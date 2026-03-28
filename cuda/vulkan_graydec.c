// Vulkan compute brute-force: gray_decode, 5-op pool, depth up to 18
// Build: gcc -O2 -o vulkan_graydec vulkan_graydec.c -lvulkan -lm
// Usage: ./vulkan_graydec [max-len]  (default 18)

#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define NUM_OPS 5
static const char *opNames[] = {"SAVE", "SHR", "XOR_B", "RLCA", "RRCA"};

static uint64_t ipow(uint64_t b, int e) { uint64_t r=1; for(int i=0;i<e;i++) r*=b; return r; }

#define VK_CHECK(x) do { VkResult r = (x); if (r != VK_SUCCESS) { fprintf(stderr, "FAIL %s = %d at line %d\n", #x, r, __LINE__); return 1; } } while(0)

int main(int argc, char **argv) {
    int maxLen = 18;
    if (argc > 1) maxLen = atoi(argv[1]);

    // Generate gray decode target
    uint8_t gray_target[256];
    for (int i = 0; i < 256; i++) {
        uint8_t x = (uint8_t)i;
        x ^= (x >> 1); x ^= (x >> 2); x ^= (x >> 4);
        gray_target[i] = x;
    }

    // Pack target into uint32[64]
    uint32_t target_packed[64];
    for (int i = 0; i < 64; i++) {
        target_packed[i] = gray_target[i*4] | (gray_target[i*4+1] << 8) |
                           (gray_target[i*4+2] << 16) | (gray_target[i*4+3] << 24);
    }

    printf("Gray decode Vulkan brute-force\n");
    printf("5-op pool: SAVE SHR XOR_B RLCA RRCA\n");
    printf("Max depth: %d\n", maxLen);

    // --- Vulkan setup ---
    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "GrayDec", 1, "BruteGPU", 1, VK_API_VERSION_1_0};
    VkInstanceCreateInfo instInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &appInfo, 0, NULL, 0, NULL};
    VkInstance inst;
    VK_CHECK(vkCreateInstance(&instInfo, NULL, &inst));

    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(inst, &devCount, NULL);
    VkPhysicalDevice *devs = malloc(devCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(inst, &devCount, devs);
    VkPhysicalDevice phys = devs[0];
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    printf("Device: %s\n", props.deviceName);

    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, NULL);
    VkQueueFamilyProperties *qfProps = malloc(qfCount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, qfProps);
    uint32_t queueFamily = 0;
    for (uint32_t i = 0; i < qfCount; i++)
        if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { queueFamily = i; break; }

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, NULL, 0, queueFamily, 1, &prio};
    VkDeviceCreateInfo devInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, NULL, 0, 1, &qInfo, 0, NULL, 0, NULL, NULL};
    VkDevice device;
    VK_CHECK(vkCreateDevice(phys, &devInfo, NULL, &device));

    VkQueue queue;
    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    // Load SPIR-V
    FILE *f = fopen("graydec_search.spv", "rb");
    if (!f) { fprintf(stderr, "FAIL: can't open graydec_search.spv\n"); return 1; }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc(sz);
    fread(code, 1, sz, f); fclose(f);
    printf("Loaded shader (%zu bytes)\n", sz);

    VkShaderModuleCreateInfo smInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sz, code};
    VkShaderModule shader;
    VK_CHECK(vkCreateShaderModule(device, &smInfo, NULL, &shader));

    // Buffer: 64 uint32 target + 3 uint32 results = 268 bytes, round to 512
    uint32_t bufSize = 512;
    VkBufferCreateInfo bufCI = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, bufSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkBuffer buffer;
    vkCreateBuffer(device, &bufCI, NULL, &buffer);

    VkMemoryRequirements memReqs;
    vkGetBufferMemoryRequirements(device, buffer, &memReqs);
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    uint32_t memIdx = 0;
    for (uint32_t i = 0; i < memProps.memoryTypeCount; i++)
        if ((memReqs.memoryTypeBits & (1 << i)) &&
            (memProps.memoryTypes[i].propertyFlags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)))
            { memIdx = i; break; }

    VkMemoryAllocateInfo allocInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, NULL, memReqs.size, memIdx};
    VkDeviceMemory mem;
    vkAllocateMemory(device, &allocInfo, NULL, &mem);
    vkBindBufferMemory(device, buffer, mem, 0);

    // Descriptor set layout
    VkDescriptorSetLayoutBinding binding = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dslInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &binding};
    VkDescriptorSetLayout dsl;
    vkCreateDescriptorSetLayout(device, &dslInfo, NULL, &dsl);

    // Push constants: seqLen, offsetLo, offsetHi, count (4 × uint32)
    VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo plInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcRange};
    VkPipelineLayout pipeLayout;
    vkCreatePipelineLayout(device, &plInfo, NULL, &pipeLayout);

    VkComputePipelineCreateInfo cpInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", NULL},
        pipeLayout, VK_NULL_HANDLE, 0};
    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeline));

    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &poolSize};
    VkDescriptorPool descPool;
    vkCreateDescriptorPool(device, &dpInfo, NULL, &descPool);

    VkDescriptorSetAllocateInfo dsaInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, descPool, 1, &dsl};
    VkDescriptorSet descSet;
    vkAllocateDescriptorSets(device, &dsaInfo, &descSet);

    VkDescriptorBufferInfo dbInfo = {buffer, 0, bufSize};
    VkWriteDescriptorSet wds = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, descSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbInfo, NULL};
    vkUpdateDescriptorSets(device, 1, &wds, 0, NULL);

    VkCommandPoolCreateInfo cmdPoolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, queueFamily};
    VkCommandPool cmdPool;
    vkCreateCommandPool(device, &cmdPoolInfo, NULL, &cmdPool);

    VkCommandBufferAllocateInfo cmdBufInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmdBuf;
    vkAllocateCommandBuffers(device, &cmdBufInfo, &cmdBuf);

    VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0};
    VkFence fence;
    vkCreateFence(device, &fenceInfo, NULL, &fence);

    // --- Search loop ---
    for (int len = 1; len <= maxLen; len++) {
        uint64_t total = ipow(NUM_OPS, len);

        // Upload target + init results
        uint32_t *mapped;
        vkMapMemory(device, mem, 0, bufSize, 0, (void**)&mapped);
        memcpy(mapped, target_packed, 64 * sizeof(uint32_t));
        mapped[64] = 0xFFFFFFFF;  // bestScore
        mapped[65] = 0;           // bestIdxLo
        mapped[66] = 0;           // bestIdxHi
        vkUnmapMemory(device, mem);

        time_t t0 = time(NULL);
        fprintf(stderr, "len=%d: searching %.2e candidates...\n", len, (double)total);

        uint32_t batchSize = 256 * 65535;  // ~16M per dispatch
        for (uint64_t off = 0; off < total; off += batchSize) {
            uint64_t cnt = total - off;
            if (cnt > batchSize) cnt = batchSize;
            uint32_t groups = (uint32_t)((cnt + 255) / 256);

            uint32_t pc[4] = {(uint32_t)len, (uint32_t)(off & 0xFFFFFFFF), (uint32_t)(off >> 32), (uint32_t)cnt};

            vkResetCommandBuffer(cmdBuf, 0);
            VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
            vkBeginCommandBuffer(cmdBuf, &beginInfo);
            vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, NULL);
            vkCmdPushConstants(cmdBuf, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
            vkCmdDispatch(cmdBuf, groups, 1, 1);
            vkEndCommandBuffer(cmdBuf);

            vkResetFences(device, 1, &fence);
            VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmdBuf, 0, NULL};
            vkQueueSubmit(queue, 1, &submitInfo, fence);
            vkWaitForFences(device, 1, &fence, VK_TRUE, 10000000000ULL);  // 10s timeout
        }

        // Read results
        vkMapMemory(device, mem, 0, bufSize, 0, (void**)&mapped);
        uint32_t score = mapped[64];
        uint64_t bestIdx = (uint64_t)mapped[66] << 32 | (uint64_t)mapped[65];
        vkUnmapMemory(device, mem);

        time_t t1 = time(NULL);

        if (score != 0xFFFFFFFF) {
            int err = score >> 16;
            int rlen = (score >> 8) & 0xFF;

            uint8_t ops[20];
            uint64_t tmp = bestIdx;
            for (int i = rlen - 1; i >= 0; i--) { ops[i] = tmp % NUM_OPS; tmp /= NUM_OPS; }

            printf("gray_dec  len=%d err=%d:", rlen, err);
            for (int i = 0; i < rlen; i++) printf(" %s", opNames[ops[i]]);
            if (err == 0) printf(" [EXACT]");
            else printf(" [approx, max_err=%d]", err);
            printf("  (%lds)\n", (long)(t1 - t0));
            fflush(stdout);

            if (err == 0) {
                printf("\n*** EXACT SOLUTION FOUND! ***\n");
                goto cleanup;
            }
        } else {
            fprintf(stderr, "  len=%d: no improvement (%lds)\n", len, (long)(t1 - t0));
        }
    }

cleanup:
    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, cmdPool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineLayout(device, pipeLayout, NULL);
    vkDestroyDescriptorPool(device, descPool, NULL);
    vkDestroyDescriptorSetLayout(device, dsl, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    vkFreeMemory(device, mem, NULL);
    vkDestroyShaderModule(device, shader, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(inst, NULL);
    return 0;
}
