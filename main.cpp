#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>
#include <fstream>

struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

static std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    size_t size = file.tellg();
    std::vector<char> buf(size);
    file.seekg(0);
    file.read(buf.data(), size);
    file.close();

    return buf;
}

class TriangleApp {
public:
    TriangleApp(uint32_t window_width, uint32_t  window_height, std::string title) {
        _window_width = window_width;
        _window_height = window_height;
        _title = title;
    }

    void run() {
        initWindow(_window_width, _window_height, _title);
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    std::string _title;
    uint32_t _window_width, _window_height;

    GLFWwindow* _window;
    vk::Instance _instance;
    vk::PhysicalDevice _physicalDevice;
    vk::Device _device;
    vk::Queue _graphicsQueue;
    uint32_t _queueFamilyIndex;
    vk::SurfaceKHR _surface;
    vk::SwapchainKHR _swapchain;
    vk::Format _swapchainFormat;
    vk::Extent2D _swapchainExtent;
    std::vector<vk::Framebuffer> _swapchainFramebuffers;
    std::vector<vk::Image> _images;
    std::vector<vk::ImageView> _imageViews;
    vk::PipelineLayout _pipelineLayout;
    vk::RenderPass _renderPass;
    vk::Pipeline _pipeline;
    vk::CommandPool _commandPool;
    std::vector<vk::CommandBuffer> _commandBuffers;

    std::vector<vk::Semaphore> _imageSemaphores;
    std::vector<vk::Semaphore> _doneSemaphores;
    std::vector<vk::Fence> _inFlightFences;

    const int MAX_FRAMES_IN_FLIGHT = 2;
    uint32_t _currentFrame = 0;
    bool _framebufferResized = false;

    #ifdef NDEBUG
        const bool enableValidationLayers = false;
    #else
        const bool enableValidationLayers = true;
    #endif
    const std::vector<const char*> validationLayers = {
        "VK_LAYER_KHRONOS_validation"
    };
    const std::vector<const char*> requiredExtensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

    void initWindow(uint32_t w, uint32_t h, std::string title) {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API); // Not OpenGL
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE); // Resizing

        _window = glfwCreateWindow(w, h, title.c_str(), nullptr, nullptr);
        glfwSetWindowUserPointer(_window, this);
        glfwSetFramebufferSizeCallback(_window, onResize);
    }

    static void onResize(GLFWwindow *window, int width, int height) {
        TriangleApp *app = (TriangleApp*) glfwGetWindowUserPointer(window);
        app->_framebufferResized = true;
    }

    void initVulkan() {
        createVulkanInstance();
        createSurface();
        getPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createPipeline();
        createFramebuffers();
        createCommandPool();
        createCommandBuffers();
        createSync();
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(_window)) {
            glfwPollEvents();
            drawFrame();
        }

        _device.waitIdle();
    }

    void cleanup() {
        cleanupSwapChain();

        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            _device.destroySemaphore(_imageSemaphores[i]);
            _device.destroySemaphore(_doneSemaphores[i]);
            _device.destroyFence(_inFlightFences[i]);
        }

        _device.destroyCommandPool(_commandPool);
        _device.destroyPipeline(_pipeline);
        _device.destroyPipelineLayout(_pipelineLayout);
        _device.destroyRenderPass(_renderPass);
        _instance.destroySurfaceKHR(_surface);
        _device.destroy();
        _instance.destroy();

        glfwDestroyWindow(_window);
        glfwTerminate();
    }

    void cleanupSwapChain() {
        for (vk::Framebuffer buf : _swapchainFramebuffers) _device.destroyFramebuffer(buf);
        for (vk::ImageView view : _imageViews) _device.destroyImageView(view);
        _device.destroySwapchainKHR(_swapchain);
    }

    void createVulkanInstance() {
        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("Unavailable validation layers requested");
        }

        vk::ApplicationInfo appInfo{};
        appInfo.pApplicationName = _title.c_str();
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0); // TODO: Move this data elsewhere
        appInfo.pEngineName = "Custom Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        vk::InstanceCreateInfo createInfo{};
        createInfo.pApplicationInfo = &appInfo;

        std::vector<const char*> extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = extensions.size();
        createInfo.ppEnabledExtensionNames = extensions.data();

        createInfo.enabledLayerCount = 0;
        if (enableValidationLayers) {
            createInfo.enabledLayerCount = validationLayers.size();
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }

        _instance = vk::createInstance(createInfo);
    }

    std::vector<const char*> getRequiredExtensions() {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        if (enableValidationLayers) {
            extensions.push_back(vk::EXTDebugUtilsExtensionName);
        }

        return extensions;
    }

    bool checkValidationLayerSupport() {
        std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

        for (const char* desiredLayer : validationLayers) {
            if (std::none_of(availableLayers.begin(), availableLayers.end(),
                [&desiredLayer](vk::LayerProperties availableLayer) {
                    return strcmp(desiredLayer, availableLayer.layerName) == 0;
                }
            )) {
                return false;
            }
        }

        return true;
    }

    void getPhysicalDevice() {
        std::vector<vk::PhysicalDevice> devices = _instance.enumeratePhysicalDevices();
        if (devices.empty()) {
            throw std::runtime_error("No physical devices with Vulkan support found");
        }

        std::vector<vk::PhysicalDevice>::iterator suitableDevice = std::find_if(devices.begin(), devices.end(),
            [this](const vk::PhysicalDevice &device) {
                vk::PhysicalDeviceProperties devProperties = device.getProperties();
                vk::PhysicalDeviceFeatures devFeatures = device.getFeatures();
                std::vector<vk::QueueFamilyProperties> families = device.getQueueFamilyProperties();
                std::vector<vk::ExtensionProperties> extensions = device.enumerateDeviceExtensionProperties();
                SwapChainSupportDetails chainDetails = swapChainSupport(device);

                int i = 0;
                return devProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu
                    && devFeatures.geometryShader
                    && std::any_of(families.begin(), families.end(),
                        [&device, &i, this](const vk::QueueFamilyProperties &family) {
                            return ((uint32_t)family.queueFlags & VK_QUEUE_GRAPHICS_BIT) && device.getSurfaceSupportKHR(i++, _surface);
                        }
                    )
                    && std::all_of(requiredExtensions.begin(), requiredExtensions.end(),
                        [&extensions](const char* req) {
                            return any_of(extensions.begin(), extensions.end(),
                                [&req](const vk::ExtensionProperties & ext){
                                    return !strcmp(ext.extensionName, req);
                                }
                            );
                        }
                    )
                    && (!chainDetails.formats.empty() && !chainDetails.presentModes.empty());
            }
        );

        if (suitableDevice == devices.end()) {
            throw std::runtime_error("No suitable physical device found");
        }

        _physicalDevice = *suitableDevice;
    }

    void createLogicalDevice() {
        int i = 0;
        std::vector<vk::QueueFamilyProperties> families = _physicalDevice.getQueueFamilyProperties();
        std::vector<vk::QueueFamilyProperties>::iterator suitableFamily = std::find_if(families.begin(), families.end(),
            [&i, this](const vk::QueueFamilyProperties &family) {
                return (uint32_t)family.queueFlags & VK_QUEUE_GRAPHICS_BIT && _physicalDevice.getSurfaceSupportKHR(i++, _surface);
            }
        );
        uint32_t suitableQueueFamilyIndex = suitableFamily - families.begin();
        _queueFamilyIndex = suitableQueueFamilyIndex;

        vk::DeviceQueueCreateInfo queueInfo{};
        queueInfo.queueFamilyIndex = suitableQueueFamilyIndex;
        queueInfo.queueCount = 1;
        float queuePriorities[] = { 1.0 };
        queueInfo.pQueuePriorities = queuePriorities;

        vk::PhysicalDeviceFeatures deviceFeatures{};

        vk::DeviceCreateInfo deviceInfo{};
        deviceInfo.queueCreateInfoCount = 1;
        deviceInfo.pQueueCreateInfos = &queueInfo;
        deviceInfo.pEnabledFeatures = &deviceFeatures;
        deviceInfo.enabledExtensionCount = requiredExtensions.size();
        deviceInfo.ppEnabledExtensionNames = requiredExtensions.data();
        deviceInfo.enabledLayerCount = 0;
        if (enableValidationLayers) {
            deviceInfo.enabledLayerCount = validationLayers.size();
            deviceInfo.ppEnabledLayerNames = validationLayers.data();
        }

        _device = _physicalDevice.createDevice(deviceInfo);
        _graphicsQueue = _device.getQueue(suitableQueueFamilyIndex, 0);
    }

    void createSurface() {
        VkSurfaceKHR tempSurface;
        if (glfwCreateWindowSurface(_instance, _window, nullptr, &tempSurface) != VK_SUCCESS) {
            throw std::runtime_error("Couldn't create window surface");
        }

        _surface = tempSurface;
    }

    SwapChainSupportDetails swapChainSupport(vk::PhysicalDevice device) {
        SwapChainSupportDetails r;

        r.capabilities = device.getSurfaceCapabilitiesKHR(_surface);
        r.formats = device.getSurfaceFormatsKHR(_surface);
        r.presentModes = device.getSurfacePresentModesKHR(_surface);

        return r;
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(std::vector<vk::SurfaceFormatKHR> &formats) {
        for (vk::SurfaceFormatKHR &f : formats) {
            if (f.format == vk::Format::eB8G8R8A8Srgb && f.colorSpace == vk::ColorSpaceKHR::eVkColorspaceSrgbNonlinear) {
                return f;
            }
        }

        return formats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(std::vector<vk::PresentModeKHR> &presentModes) {
        return std::any_of(presentModes.begin(), presentModes.end(),
            [](const vk::PresentModeKHR &mode) {
                return mode == vk::PresentModeKHR::eMailbox;
            }
        ) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }

        uint32_t width, height;
        glfwGetFramebufferSize(_window, (int*) &width, (int*) &height);

        VkExtent2D actualExtent = {
            std::min(std::max(width, capabilities.minImageExtent.width), capabilities.maxImageExtent.width),
            std::min(std::max(height, capabilities.minImageExtent.height), capabilities.maxImageExtent.height),
        };

        return actualExtent;
    }

    void createSwapChain() {
        SwapChainSupportDetails support = swapChainSupport(_physicalDevice);
        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(support.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(support.presentModes);
        vk::Extent2D extent = chooseSwapExtent(support.capabilities);

        uint32_t imageCount = support.capabilities.maxImageCount > support.capabilities.minImageCount ? support.capabilities.minImageCount + 1 : support.capabilities.minImageCount;

        vk::SwapchainCreateInfoKHR createInfo{};
        createInfo.surface = _surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
        createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        createInfo.preTransform = support.capabilities.currentTransform;
        createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        createInfo.presentMode = presentMode;
        createInfo.clipped = true;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        _swapchain = _device.createSwapchainKHR(createInfo);
        _swapchainFormat = surfaceFormat.format;
        _swapchainExtent = extent;
        _images = _device.getSwapchainImagesKHR(_swapchain);
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(_window, &width, &height);
        while (!width || !height) {
            glfwGetFramebufferSize(_window, &width, &height);
            glfwWaitEvents();
        }

        _device.waitIdle();

        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createFramebuffers();
    }

    void createImageViews() {
        _imageViews.clear();
        for (vk::Image img : _images) {
            vk::ImageViewCreateInfo viewInfo{};
            viewInfo.image = img;
            viewInfo.viewType = vk::ImageViewType::e2D;
            viewInfo.format = _swapchainFormat;

            viewInfo.components.r = vk::ComponentSwizzle::eIdentity;
            viewInfo.components.g = vk::ComponentSwizzle::eIdentity;
            viewInfo.components.b = vk::ComponentSwizzle::eIdentity;
            viewInfo.components.a = vk::ComponentSwizzle::eIdentity;

            viewInfo.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
            viewInfo.subresourceRange.baseMipLevel = 0;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = 0;
            viewInfo.subresourceRange.layerCount = 1;

            _imageViews.push_back(_device.createImageView(viewInfo));
        }
    }

    void createPipeline() {
        std::vector<char> fshCode = readFile("shaders/triangle_frag.spv");
        std::vector<char> vshCode = readFile("shaders/triangle_vert.spv");

        vk::ShaderModule fshModule = createShaderModule(fshCode);
        vk::ShaderModule vshModule = createShaderModule(vshCode);

        vk::PipelineShaderStageCreateInfo fshStageInfo{};
        fshStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
        fshStageInfo.module = fshModule;
        fshStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo vshStageInfo{};
        vshStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
        vshStageInfo.module = vshModule;
        vshStageInfo.pName = "main";

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vshStageInfo, fshStageInfo};

        std::vector<vk::DynamicState> dynamicStates = { vk::DynamicState::eViewport, vk::DynamicState::eScissor };
        vk::PipelineDynamicStateCreateInfo dynamicStateInfo{};
        dynamicStateInfo.dynamicStateCount = dynamicStates.size();
        dynamicStateInfo.pDynamicStates = dynamicStates.data();

        vk::PipelineViewportStateCreateInfo viewportStateInfo{};
        viewportStateInfo.viewportCount = 1;
        viewportStateInfo.scissorCount = 1;

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
        vertexInputInfo.vertexBindingDescriptionCount = 0;
        vertexInputInfo.vertexAttributeDescriptionCount = 0;

        vk::PipelineInputAssemblyStateCreateInfo inputAssemblyInfo{};
        inputAssemblyInfo.topology = vk::PrimitiveTopology::eTriangleList;
        inputAssemblyInfo.primitiveRestartEnable = VK_FALSE;

        vk::PipelineRasterizationStateCreateInfo rasterInfo{};
        rasterInfo.depthClampEnable = false;
        rasterInfo.rasterizerDiscardEnable = false;
        rasterInfo.lineWidth = 1.0f;
        rasterInfo.cullMode = vk::CullModeFlagBits::eBack;
        rasterInfo.frontFace = vk::FrontFace::eClockwise;
        rasterInfo.depthBiasEnable = false;

        vk::PipelineMultisampleStateCreateInfo multisampleInfo{};
        multisampleInfo.sampleShadingEnable = false;
        multisampleInfo.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineColorBlendAttachmentState colorBlendInfo{};
        colorBlendInfo.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;
        colorBlendInfo.blendEnable = false;

        vk::PipelineColorBlendStateCreateInfo blendingInfo{};
        blendingInfo.logicOpEnable = false;
        blendingInfo.attachmentCount = 1;
        blendingInfo.pAttachments = &colorBlendInfo;

        vk::PipelineLayoutCreateInfo layoutInfo{};
        _pipelineLayout = _device.createPipelineLayout(layoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssemblyInfo;
        pipelineInfo.pViewportState = &viewportStateInfo;
        pipelineInfo.pRasterizationState = &rasterInfo;
        pipelineInfo.pMultisampleState = &multisampleInfo;
        pipelineInfo.pColorBlendState = &blendingInfo;
        pipelineInfo.pDynamicState = &dynamicStateInfo;
        pipelineInfo.layout = _pipelineLayout;
        pipelineInfo.renderPass = _renderPass;
        pipelineInfo.subpass = 0;

        std::vector<vk::GraphicsPipelineCreateInfo> pipelineInfos = { pipelineInfo };
        vk::ResultValue<std::vector<vk::Pipeline>> r = _device.createGraphicsPipelines(nullptr, pipelineInfos);
        if (r.result != vk::Result::eSuccess) {
            throw std::runtime_error("Couldn't create graphics pipeline");
        }
        _pipeline = r.value.front();

        _device.destroyShaderModule(fshModule);
        _device.destroyShaderModule(vshModule);
    }

    vk::ShaderModule createShaderModule(const std::vector<char> &code) {
        vk::ShaderModuleCreateInfo shaderInfo{};
        shaderInfo.codeSize = code.size();
        shaderInfo.pCode = (const uint32_t*) code.data();
        return _device.createShaderModule(shaderInfo);
    }

    void createRenderPass() {
        vk::AttachmentDescription colorAttachment{};
        colorAttachment.format = _swapchainFormat;
        colorAttachment.samples = vk::SampleCountFlagBits::e1;
        colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
        colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
        colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
        colorAttachment.finalLayout = vk::ImageLayout::ePresentSrcKHR;

        vk::AttachmentReference attachmentReference{};
        attachmentReference.attachment = 0;
        attachmentReference.layout = vk::ImageLayout::eColorAttachmentOptimal;

        vk::SubpassDependency dep{};
        dep.srcSubpass = vk::SubpassExternal;
        dep.dstSubpass = 0;
        dep.srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::AccessFlags accessFlag;
        dep.srcAccessMask = accessFlag;
        dep.dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        dep.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;

        vk::SubpassDescription subpass{};
        subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &attachmentReference;

        vk::RenderPassCreateInfo renderPassInfo{};
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.setDependencies(dep);

        _renderPass = _device.createRenderPass(renderPassInfo);
    }

    void createFramebuffers() {
        _swapchainFramebuffers.clear();
        for (vk::ImageView view : _imageViews) {

            vk::FramebufferCreateInfo framebufferInfo{};
            framebufferInfo.renderPass = _renderPass;
            framebufferInfo.attachmentCount = 1;
            framebufferInfo.pAttachments = &view;
            framebufferInfo.width = _swapchainExtent.width;
            framebufferInfo.height = _swapchainExtent.height;
            framebufferInfo.layers = 1;

            _swapchainFramebuffers.push_back(_device.createFramebuffer(framebufferInfo));
        }
    }

    void createCommandPool() {
        vk::CommandPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        poolInfo.queueFamilyIndex = _queueFamilyIndex;
        _commandPool = _device.createCommandPool(poolInfo);
    }

    void createCommandBuffers() {
        vk::CommandBufferAllocateInfo commandBufferInfo{};
        commandBufferInfo.commandPool = _commandPool;
        commandBufferInfo.level = vk::CommandBufferLevel::ePrimary;
        commandBufferInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
        _commandBuffers = _device.allocateCommandBuffers(commandBufferInfo);
    }

    void recordCommands(vk::CommandBuffer buf, uint32_t imageIndex) {
        vk::CommandBufferBeginInfo beginInfo{};
        buf.begin(beginInfo);

        vk::RenderPassBeginInfo renderPassInfo{};
        renderPassInfo.renderPass = _renderPass;
        renderPassInfo.framebuffer = _swapchainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset.setX(0).setY(0);
        renderPassInfo.renderArea.extent = _swapchainExtent;
        vk::ClearColorValue clearColor(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
        vk::ClearDepthStencilValue clearDepth(0.0f, 0.0f);
        vk::ClearValue clearCol;
        clearCol.setColor(clearColor).setDepthStencil(clearDepth);
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearCol;
        buf.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

        buf.bindPipeline(vk::PipelineBindPoint::eGraphics, _pipeline);

        vk::Viewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float) _swapchainExtent.width;
        viewport.height = (float) _swapchainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        buf.setViewport(0, viewport);

        vk::Rect2D scissor{};
        scissor.offset.setX(0).setY(0);
        scissor.extent = _swapchainExtent;
        buf.setScissor(0, scissor);

        buf.draw(3, 1, 0, 0);

        buf.endRenderPass();
        buf.end();
    }

    void createSync() {
        for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vk::SemaphoreCreateInfo semaphoreInfo{};
            _imageSemaphores.push_back(_device.createSemaphore(semaphoreInfo));
            _doneSemaphores.push_back(_device.createSemaphore(semaphoreInfo));

            vk::FenceCreateInfo fenceInfo{};
            fenceInfo.flags = vk::FenceCreateFlagBits::eSignaled;
            _inFlightFences.push_back(_device.createFence(fenceInfo));
        }
    }

    void drawFrame() {
        _device.waitForFences(_inFlightFences[_currentFrame], true, UINT64_MAX);

        vk::ResultValue<uint32_t> r = _device.acquireNextImageKHR(_swapchain, UINT64_MAX, _imageSemaphores[_currentFrame]);
        if (r.result == vk::Result::eErrorOutOfDateKHR) {
            recreateSwapChain();
            return;
        }
        if (r.result != vk::Result::eSuccess && r.result != vk::Result::eSuboptimalKHR) {
            throw std::runtime_error("Couldn't acquire image from swapchain");
        }
        uint32_t imageIndex = r.value;
        _device.resetFences(_inFlightFences[_currentFrame]);

        _commandBuffers[_currentFrame].reset();
        recordCommands(_commandBuffers[_currentFrame], imageIndex);

        vk::SubmitInfo submitInfo{};
        submitInfo.setWaitSemaphores(_imageSemaphores[_currentFrame]);
        vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::PipelineStageFlags stageFlags = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        submitInfo.setWaitDstStageMask(stageFlags);
        submitInfo.setCommandBuffers(_commandBuffers[_currentFrame]);
        submitInfo.setSignalSemaphores(_doneSemaphores[_currentFrame]);
        _graphicsQueue.submit(submitInfo, _inFlightFences[_currentFrame]);

        vk::PresentInfoKHR presentInfo{};
        presentInfo.setWaitSemaphores(_doneSemaphores[_currentFrame]);
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = &_swapchain;
        presentInfo.pImageIndices = &imageIndex;
        try {
            vk::Result res = _graphicsQueue.presentKHR(presentInfo);
            if (res == vk::Result::eErrorOutOfDateKHR || res == vk::Result::eSuboptimalKHR || _framebufferResized) {
                _framebufferResized = false;
                recreateSwapChain();
            }
            else if (res != vk::Result::eSuccess) {
                throw std::runtime_error("Couldn't present image from swapchain");
            }
        }
        catch (vk::OutOfDateKHRError e) {
            _framebufferResized = false;
            recreateSwapChain();
        }

        _currentFrame = (_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
};

int main() {
    TriangleApp app(1280, 720, "Triangle");

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

	return EXIT_SUCCESS;
}