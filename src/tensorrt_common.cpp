// Copyright 2020 Tier IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <tensorrt_common/tensorrt_common.hpp>

#include <NvInferPlugin.h>

#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace tensorrt_common
{

TrtCommon::TrtCommon(
  const std::string & model_path, const std::string & precision,
  std::unique_ptr<nvinfer1::IInt8EntropyCalibrator2> calibrator,
  const tensorrt_common::BatchConfig & batch_config,
  const size_t max_workspace_size)
: model_file_path_(model_path),
  calibrator_(std::move(calibrator)),
  precision_(precision),
  batch_config_(batch_config),
  max_workspace_size_(max_workspace_size)
{
  runtime_ = TrtUniquePtr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger_));
  initLibNvInferPlugins(&logger_, "");
}

void TrtCommon::setup()
{
  if (!std::filesystem::exists(model_file_path_)) {
    is_initialized_ = false;
    return;
  }
  if (model_file_path_.extension() == ".engine") {
    loadEngine(model_file_path_);
  } else if (model_file_path_.extension() == ".onnx") {
    std::filesystem::path cache_engine_path{model_file_path_};
    cache_engine_path.replace_extension("engine");
    if (std::filesystem::exists(cache_engine_path)) {
      loadEngine(cache_engine_path);
    } else {
      logger_.log(nvinfer1::ILogger::Severity::kINFO, "Start build engine");
      buildEngineFromOnnx(model_file_path_, cache_engine_path);
      logger_.log(nvinfer1::ILogger::Severity::kINFO, "End build engine");
    }
  } else {
    is_initialized_ = false;
    return;
  }

  context_ = TrtUniquePtr<nvinfer1::IExecutionContext>(engine_->createExecutionContext());
  if (!context_) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create context");
    is_initialized_ = false;
    return;
  }

  is_initialized_ = true;
}

bool TrtCommon::loadEngine(const std::string & engine_file_path)
{
  std::ifstream engine_file(engine_file_path);
  std::stringstream engine_buffer;
  engine_buffer << engine_file.rdbuf();
  std::string engine_str = engine_buffer.str();
  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(
      reinterpret_cast<const void *>(engine_str.data()), engine_str.size()));
  return true;
}

bool TrtCommon::buildEngineFromOnnx(
  const std::string & onnx_file_path,
  const std::string & output_engine_file_path)
{
  auto builder = TrtUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger_));
  if (!builder) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create builder");
    return false;
  }

  const auto explicitBatch =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  auto network =
    TrtUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
  if (!network) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create network");
    return false;
  }

  auto config = TrtUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
  if (!config) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create builder config");
    return false;
  }

  if (precision_ == "fp16" || precision_ == "int8") {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  config->setMaxWorkspaceSize(max_workspace_size_);

  auto parser = TrtUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger_));
  if (!parser->parseFromFile(
      onnx_file_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR)))
  {
    return false;
  }

  const auto input = network->getInput(0);
  const auto input_dims = input->getDimensions();
  const auto input_channel = input_dims.d[1];
  const auto input_height = input_dims.d[2];
  const auto input_width = input_dims.d[3];

  auto profile = builder->createOptimizationProfile();
  profile->setDimensions(
    network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMIN,
    nvinfer1::Dims4{batch_config_.at(0), input_channel, input_height, input_width});
  profile->setDimensions(
    network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kOPT,
    nvinfer1::Dims4{batch_config_.at(1), input_channel, input_height, input_width});
  profile->setDimensions(
    network->getInput(0)->getName(), nvinfer1::OptProfileSelector::kMAX,
    nvinfer1::Dims4{batch_config_.at(2), input_channel, input_height, input_width});
  config->addOptimizationProfile(profile);

  if (precision_ == "int8" && calibrator_) {
    config->setFlag(nvinfer1::BuilderFlag::kINT8);
    config->setInt8Calibrator(calibrator_.get());
  }

#if TENSORRT_VERSION_MAJOR >= 8
  auto plan =
    TrtUniquePtr<nvinfer1::IHostMemory>(builder->buildSerializedNetwork(*network, *config));
  if (!plan) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create host memory");
    return false;
  }
  engine_ =
    TrtUniquePtr<nvinfer1::ICudaEngine>(
    runtime_->deserializeCudaEngine(
      plan->data(),
      plan->size()));
#else
  engine_ = TrtUniquePtr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
#endif

  if (!engine_) {
    logger_.log(nvinfer1::ILogger::Severity::kERROR, "Fail to create engine");
    return false;
  }

  // save engine
#if TENSORRT_VERSION_MAJOR < 8
  auto data = TrtUniquePtr<nvinfer1::IHostMemory>(engine_->serialize());
#endif
  std::ofstream file;
  file.open(output_engine_file_path, std::ios::binary | std::ios::out);
  if (!file.is_open()) {
    return false;
  }
#if TENSORRT_VERSION_MAJOR < 8
  file.write(reinterpret_cast<const char *>(data->data()), data->size());
#else
  file.write(reinterpret_cast<const char *>(plan->data()), plan->size());
#endif

  file.close();

  return true;
}

bool TrtCommon::isInitialized() {return is_initialized_;}

nvinfer1::Dims TrtCommon::getBindingDimensions(const int32_t index) const
{
  return context_->getBindingDimensions(index);
}


bool TrtCommon::setBindingDimensions(const int32_t index, const nvinfer1::Dims & dimensions) const
{
  return context_->setBindingDimensions(index, dimensions);
}

bool TrtCommon::enqueueV2(
  void * const * bindings, cudaStream_t stream,
  cudaEvent_t * input_consumed)
{
  return context_->enqueueV2(bindings, stream, input_consumed);
}

}  // namespace tensorrt_common
