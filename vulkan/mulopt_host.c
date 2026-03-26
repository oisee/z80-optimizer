// mulopt_host.c — Minimal Vulkan compute host for mulopt search
//
// Build: gcc -O2 -o vk_mulopt vulkan/mulopt_host.c -lvulkan -lm
// Usage: vk_mulopt [--max-len 8] [--k 42] [--json]
//
// Loads gen_z80.spv from current directory.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vulkan/vulkan.h>

static int NUM_OPS = 14;
#define MAX_LEN 12

static const char *opNames[] = {
    "ADD A,A", "ADD A,B", "SUB B", "LD B,A",
    "ADC A,B", "ADC A,A", "SBC A,B", "SBC A,A",
    "SRL A", "RLA", "RRA", "RLCA", "RRCA", "NEG"
};
static const uint8_t opCosts[] = {4,4,4,4,4,4,4,4,8,4,4,4,4,8};

static uint64_t ipow(uint64_t base, int exp) {
    uint64_t r = 1;
    for (int i = 0; i < exp; i++) r *= base;
    return r;
}

static void decode_seq(uint64_t idx, int len, uint8_t *ops) {
    for (int i = len - 1; i >= 0; i--) {
        ops[i] = (uint8_t)(idx % NUM_OPS);
        idx /= NUM_OPS;
    }
}

static uint8_t cpu_run_seq(uint8_t *ops, int len, uint8_t input) {
    uint8_t a = input, b = 0;
    int carry = 0;
    for (int i = 0; i < len; i++) {
        uint16_t r; uint8_t bit, c;
        switch (ops[i]) {
        case 0:  r=a+a; carry=r>0xFF; a=(uint8_t)r; break;
        case 1:  r=a+b; carry=r>0xFF; a=(uint8_t)r; break;
        case 2:  carry=(a<b); a=a-b; break;
        case 3:  b=a; break;
        case 4:  c=carry?1:0; r=a+b+c; carry=r>0xFF; a=(uint8_t)r; break;
        case 5:  c=carry?1:0; r=a+a+c; carry=r>0xFF; a=(uint8_t)r; break;
        case 6:  c=carry?1:0; carry=((int)a-(int)b-c)<0; a=a-b-c; break;
        case 7:  c=carry?1:0; carry=c>0; a=-(uint8_t)c; break;
        case 8:  carry=a&1; a=a>>1; break;
        case 9:  bit=carry?1:0; carry=(a&0x80)!=0; a=(a<<1)|bit; break;
        case 10: bit=carry?0x80:0; carry=a&1; a=(a>>1)|bit; break;
        case 11: carry=(a&0x80)!=0; a=(a<<1)|(a>>7); break;
        case 12: carry=a&1; a=(a>>1)|(a<<7); break;
        case 13: carry=(a!=0); a=(uint8_t)(0-a); break;
        }
    }
    return a;
}

#define VK_CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { fprintf(stderr, "Vulkan error %d at %s:%d\n", r, __FILE__, __LINE__); exit(1); } } while(0)

// SSBO layout matching the shader
typedef struct {
    uint32_t k;
    int32_t seqLen;
    uint64_t offset;
    uint64_t count;
} ArgsSSBO;

typedef struct {
    uint32_t bestScore;
    uint32_t bestIdxLo;
    uint32_t bestIdxHi;
} ResultSSBO;

static uint32_t *readSPV(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    *size = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint32_t *buf = malloc(*size);
    fread(buf, 1, *size, f);
    fclose(f);
    return buf;
}

int main(int argc, char *argv[]) {
    int maxLen = 8, singleK = 0, jsonMode = 0, skipCpuVerify = 0, numOps = 0;
    const char *spvPath = "gen_z80.spv";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--max-len") && i+1 < argc) maxLen = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i+1 < argc) singleK = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--json")) jsonMode = 1;
        else if (!strcmp(argv[i], "--spv") && i+1 < argc) spvPath = argv[++i];
        else if (!strcmp(argv[i], "--no-verify")) skipCpuVerify = 1;
        else if (!strcmp(argv[i], "--num-ops") && i+1 < argc) { numOps = atoi(argv[++i]); NUM_OPS = numOps; }
    }

    // Create instance
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "mulopt", 1, "gpugen", 1, VK_API_VERSION_1_3};
    VkInstanceCreateInfo instInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &appInfo, 0, NULL, 0, NULL};
    VkInstance instance;
    VK_CHECK(vkCreateInstance(&instInfo, NULL, &instance));

    // Pick first GPU
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance, &devCount, NULL);
    if (devCount == 0) { fprintf(stderr, "No Vulkan devices\n"); return 1; }
    VkPhysicalDevice *devs = malloc(devCount * sizeof(VkPhysicalDevice));
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &devCount, devs));
    VkPhysicalDevice physDev = devs[0];
    free(devs);

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physDev, &props);
    fprintf(stderr, "Vulkan device: %s\n", props.deviceName);

    // Find compute queue family
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, NULL);
    VkQueueFamilyProperties *qfProps = malloc(qfCount * sizeof(*qfProps));
    vkGetPhysicalDeviceQueueFamilyProperties(physDev, &qfCount, qfProps);
    uint32_t computeQF = 0;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { computeQF = i; break; }
    }
    free(qfProps);

    // Create device
    float queuePrio = 1.0f;
    VkDeviceQueueCreateInfo queueInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, computeQF, 1, &queuePrio};
    VkDeviceCreateInfo devInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &queueInfo, 0, NULL, 0, NULL, NULL};
    VkDevice device;
    VK_CHECK(vkCreateDevice(physDev, &devInfo, NULL, &device));

    VkQueue queue;
    vkGetDeviceQueue(device, computeQF, 0, &queue);

    // Load shader
    size_t spvSize;
    uint32_t *spvCode = readSPV(spvPath, &spvSize);
    VkShaderModuleCreateInfo smInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, spvSize, spvCode};
    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &shaderModule));
    free(spvCode);

    // Descriptor set layout: 2 bindings (args SSBO, result SSBO)
    VkDescriptorSetLayoutBinding bindings[2] = {
        {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
    };
    VkDescriptorSetLayoutCreateInfo dslInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 2, bindings};
    VkDescriptorSetLayout dsLayout;
    VK_CHECK(vkCreateDescriptorSetLayout(device, &dslInfo, NULL, &dsLayout));

    // Pipeline layout
    VkPipelineLayoutCreateInfo plInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsLayout, 0, NULL};
    VkPipelineLayout pipeLayout;
    VK_CHECK(vkCreatePipelineLayout(device, &plInfo, NULL, &pipeLayout));

    // Compute pipeline
    VkComputePipelineCreateInfo cpInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, shaderModule, "main", NULL},
        pipeLayout, VK_NULL_HANDLE, 0};
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeline));

    // Memory helpers
    uint32_t findMemType(VkPhysicalDevice pd, uint32_t typeBits, VkMemoryPropertyFlags flags) {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(pd, &memProps);
        for (uint32_t i = 0; i < memProps.memoryTypeCount; i++) {
            if ((typeBits & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & flags) == flags) return i;
        }
        return 0;
    }

    // Create buffers
    VkBufferCreateInfo bufInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, sizeof(ArgsSSBO),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkBuffer argsBuf, resultBuf;
    VK_CHECK(vkCreateBuffer(device, &bufInfo, NULL, &argsBuf));
    bufInfo.size = sizeof(ResultSSBO);
    VK_CHECK(vkCreateBuffer(device, &bufInfo, NULL, &resultBuf));

    VkMemoryRequirements memReq;
    VkMemoryPropertyFlags memFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    vkGetBufferMemoryRequirements(device, argsBuf, &memReq);
    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, memReq.size,
        findMemType(physDev, memReq.memoryTypeBits, memFlags)};
    VkDeviceMemory argsMem;
    VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL, &argsMem));
    VK_CHECK(vkBindBufferMemory(device, argsBuf, argsMem, 0));

    vkGetBufferMemoryRequirements(device, resultBuf, &memReq);
    allocInfo.allocationSize = memReq.size;
    allocInfo.memoryTypeIndex = findMemType(physDev, memReq.memoryTypeBits, memFlags);
    VkDeviceMemory resultMem;
    VK_CHECK(vkAllocateMemory(device, &allocInfo, NULL, &resultMem));
    VK_CHECK(vkBindBufferMemory(device, resultBuf, resultMem, 0));

    // Descriptor pool + set
    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &poolSize};
    VkDescriptorPool descPool;
    VK_CHECK(vkCreateDescriptorPool(device, &dpInfo, NULL, &descPool));

    VkDescriptorSetAllocateInfo dsAllocInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, descPool, 1, &dsLayout};
    VkDescriptorSet descSet;
    VK_CHECK(vkAllocateDescriptorSets(device, &dsAllocInfo, &descSet));

    VkDescriptorBufferInfo dbi0 = {argsBuf, 0, sizeof(ArgsSSBO)};
    VkDescriptorBufferInfo dbi1 = {resultBuf, 0, sizeof(ResultSSBO)};
    VkWriteDescriptorSet writes[2] = {
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, descSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi0, NULL},
        {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, descSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbi1, NULL},
    };
    vkUpdateDescriptorSets(device, 2, writes, 0, NULL);

    // Command buffer
    VkCommandPoolCreateInfo cpoolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL,
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, computeQF};
    VkCommandPool cmdPool;
    VK_CHECK(vkCreateCommandPool(device, &cpoolInfo, NULL, &cmdPool));

    VkCommandBufferAllocateInfo cbAllocInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL,
        cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmdBuf;
    VK_CHECK(vkAllocateCommandBuffers(device, &cbAllocInfo, &cmdBuf));

    VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0};
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fenceInfo, NULL, &fence));

    // Search loop
    int startK = singleK > 0 ? singleK : 2;
    int endK = singleK > 0 ? singleK : 255;
    int solved = 0;
    if (jsonMode) printf("[\n");

    for (int k = startK; k <= endK; k++) {
        int found = 0;
        int foundLen = 0, foundCost = 0;
        uint8_t foundOps[MAX_LEN];

        for (int len = 1; len <= maxLen && !found; len++) {
            uint64_t total = ipow(NUM_OPS, len);

            // Init result
            ResultSSBO *res;
            vkMapMemory(device, resultMem, 0, sizeof(ResultSSBO), 0, (void **)&res);
            res->bestScore = 0xFFFFFFFF;
            res->bestIdxLo = 0;
            res->bestIdxHi = 0;
            vkUnmapMemory(device, resultMem);

            uint64_t batchSize = 256 * 65535;
            uint64_t offset = 0;
            while (offset < total) {
                uint64_t count = total - offset;
                if (count > batchSize) count = batchSize;

                // Set args
                ArgsSSBO *args;
                vkMapMemory(device, argsMem, 0, sizeof(ArgsSSBO), 0, (void **)&args);
                args->k = k;
                args->seqLen = len;
                args->offset = offset;
                args->count = count;
                vkUnmapMemory(device, argsMem);

                // Record + submit
                VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL,
                    VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
                vkBeginCommandBuffer(cmdBuf, &beginInfo);
                vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
                vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, NULL);
                uint32_t groupCount = (uint32_t)((count + 255) / 256);
                vkCmdDispatch(cmdBuf, groupCount, 1, 1);
                vkEndCommandBuffer(cmdBuf);

                VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmdBuf, 0, NULL};
                VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
                VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
                vkResetFences(device, 1, &fence);
                vkResetCommandBuffer(cmdBuf, 0);

                offset += count;
            }

            // Read result
            vkMapMemory(device, resultMem, 0, sizeof(ResultSSBO), 0, (void **)&res);
            uint32_t bestScore = res->bestScore;
            uint64_t bestIdx = ((uint64_t)res->bestIdxHi << 32) | res->bestIdxLo;
            vkUnmapMemory(device, resultMem);

            if (bestScore != 0xFFFFFFFF) {
                found = 1;
                foundLen = len;
                foundCost = bestScore & 0xFFFF;
                decode_seq(bestIdx, len, foundOps);

                // CPU verify (only for 14-op z80_mul, skip for other ISAs)
                if (!skipCpuVerify && NUM_OPS <= 14) {
                    int ok = 1;
                    for (int inp = 0; inp < 256 && ok; inp++) {
                        if (cpu_run_seq(foundOps, len, (uint8_t)inp) != (uint8_t)(inp * k))
                            ok = 0;
                    }
                    if (!ok) {
                        fprintf(stderr, "WARNING: Vulkan result for x%d failed CPU verify!\n", k);
                        found = 0;
                    }
                }
            }
        }

        if (found) {
            solved++;
            if (jsonMode) {
                printf("  {\"k\": %d, \"ops\": [", k);
                for (int i = 0; i < foundLen; i++)
                    printf("%s\"%s\"", i ? "," : "", (foundOps[i] < (int)(sizeof(opNames)/sizeof(opNames[0])) ? opNames[foundOps[i]] : "?"));
                printf("], \"length\": %d, \"tstates\": %d}%s\n",
                       foundLen, foundCost, (k < endK) ? "," : "");
            } else {
                printf("x%d:", k);
                for (int i = 0; i < foundLen; i++) printf(" %s", (foundOps[i] < (int)(sizeof(opNames)/sizeof(opNames[0])) ? opNames[foundOps[i]] : "?"));
                printf(" (%d insts, %dT)\n", foundLen, foundCost);
            }
        }
        if (!singleK)
            fprintf(stderr, "\rx%d/%d (%d solved)...", k, endK, solved);
    }
    fprintf(stderr, "\nDone: %d/%d constants solved\n", solved, endK - startK + 1);
    if (jsonMode) printf("]\n");

    // Cleanup
    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, cmdPool, NULL);
    vkDestroyDescriptorPool(device, descPool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineLayout(device, pipeLayout, NULL);
    vkDestroyDescriptorSetLayout(device, dsLayout, NULL);
    vkDestroyShaderModule(device, shaderModule, NULL);
    vkFreeMemory(device, argsMem, NULL);
    vkFreeMemory(device, resultMem, NULL);
    vkDestroyBuffer(device, argsBuf, NULL);
    vkDestroyBuffer(device, resultBuf, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);
    return 0;
}
