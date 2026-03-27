// Minimal Vulkan compute dispatch for testing Nanz-compiled shaders
// Build: gcc -O2 -o vulkan_dispatch vulkan_dispatch.c -lvulkan
// Usage: ./vulkan_dispatch /path/to/shader.spv
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    const char *spvPath = argc > 1 ? argv[1] : "/tmp/minz_vulkan_test.spv";

    VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO, NULL, "NanzTest", 1, "NanzGPU", 1, VK_API_VERSION_1_0};
    VkInstanceCreateInfo instInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, NULL, 0, &appInfo, 0, NULL, 0, NULL};
    VkInstance inst;
    if (vkCreateInstance(&instInfo, NULL, &inst) != VK_SUCCESS) { printf("FAIL: instance\n"); return 1; }

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
    if (vkCreateDevice(phys, &devInfo, NULL, &device) != VK_SUCCESS) { printf("FAIL: device\n"); return 1; }

    VkQueue queue;
    vkGetDeviceQueue(device, queueFamily, 0, &queue);

    FILE *f = fopen(spvPath, "rb");
    if (!f) { printf("FAIL: can't open %s\n", spvPath); return 1; }
    fseek(f, 0, SEEK_END); size_t sz = ftell(f); fseek(f, 0, SEEK_SET);
    uint32_t *code = malloc(sz);
    fread(code, 1, sz, f); fclose(f);
    printf("Loaded %s (%zu bytes)\n", spvPath, sz);

    VkShaderModuleCreateInfo smInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, NULL, 0, sz, code};
    VkShaderModule shader;
    if (vkCreateShaderModule(device, &smInfo, NULL, &shader) != VK_SUCCESS) { printf("FAIL: shader\n"); return 1; }

    VkBufferCreateInfo bufInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, NULL, 0, 1024, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL};
    VkBuffer buffer;
    vkCreateBuffer(device, &bufInfo, NULL, &buffer);

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

    VkDescriptorSetLayoutBinding binding = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
    VkDescriptorSetLayoutCreateInfo dslInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, NULL, 0, 1, &binding};
    VkDescriptorSetLayout dsl;
    vkCreateDescriptorSetLayout(device, &dslInfo, NULL, &dsl);

    VkPushConstantRange pcRange = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 4};
    VkPipelineLayoutCreateInfo plInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, NULL, 0, 1, &dsl, 1, &pcRange};
    VkPipelineLayout pipeLayout;
    vkCreatePipelineLayout(device, &plInfo, NULL, &pipeLayout);

    VkComputePipelineCreateInfo cpInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, NULL, 0,
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, NULL, 0, VK_SHADER_STAGE_COMPUTE_BIT, shader, "main", NULL},
        pipeLayout, VK_NULL_HANDLE, 0};
    VkPipeline pipeline;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpInfo, NULL, &pipeline) != VK_SUCCESS) { printf("FAIL: pipeline\n"); return 1; }

    VkDescriptorPoolSize poolSize = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo dpInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, NULL, 0, 1, 1, &poolSize};
    VkDescriptorPool descPool;
    vkCreateDescriptorPool(device, &dpInfo, NULL, &descPool);

    VkDescriptorSetAllocateInfo dsaInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, NULL, descPool, 1, &dsl};
    VkDescriptorSet descSet;
    vkAllocateDescriptorSets(device, &dsaInfo, &descSet);

    VkDescriptorBufferInfo dbInfo = {buffer, 0, 1024};
    VkWriteDescriptorSet wds = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, NULL, descSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, NULL, &dbInfo, NULL};
    vkUpdateDescriptorSets(device, 1, &wds, 0, NULL);

    VkCommandPoolCreateInfo cmdPoolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, NULL, 0, queueFamily};
    VkCommandPool cmdPool;
    vkCreateCommandPool(device, &cmdPoolInfo, NULL, &cmdPool);

    VkCommandBufferAllocateInfo cmdBufInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, NULL, cmdPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1};
    VkCommandBuffer cmdBuf;
    vkAllocateCommandBuffers(device, &cmdBufInfo, &cmdBuf);

    VkCommandBufferBeginInfo beginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, NULL, VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT, NULL};
    vkBeginCommandBuffer(cmdBuf, &beginInfo);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, NULL);
    uint32_t n = 256;
    vkCmdPushConstants(cmdBuf, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 4, &n);
    vkCmdDispatch(cmdBuf, 1, 1, 1);
    vkEndCommandBuffer(cmdBuf);

    VkFenceCreateInfo fenceInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, NULL, 0};
    VkFence fence;
    vkCreateFence(device, &fenceInfo, NULL, &fence);

    VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO, NULL, 0, NULL, NULL, 1, &cmdBuf, 0, NULL};
    vkQueueSubmit(queue, 1, &submitInfo, fence);
    vkWaitForFences(device, 1, &fence, VK_TRUE, 1000000000ULL);

    uint32_t *mapped;
    vkMapMemory(device, mem, 0, 1024, 0, (void**)&mapped);
    int correct = 0;
    for (int i = 0; i < 256; i++) {
        uint32_t expected = (i * 2) & 0xFF;
        if (mapped[i] == expected) correct++;
        else if (correct < 260) printf("  MISMATCH[%d]: got %u, expected %u\n", i, mapped[i], expected);
    }
    printf("Results: %d/256 correct\n", correct);
    vkUnmapMemory(device, mem);

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
    return correct == 256 ? 0 : 1;
}
