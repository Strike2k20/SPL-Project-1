#include <fstream>
#include <stdexcept>
#include "json.hpp"

#include <iostream>
#include "Server.h"

using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for(auto& gpu: config["gpus"]){
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for(auto& model: config["models"]){
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for(auto& request: config["requests"]){
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if(typeStr != "generative" && typeStr != "text_embedder"){
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr=="generative"?RequestType::Generative:RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0) {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0) {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0) {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}

bool Server::step() {
    if(nonSubmittedRequests.empty() && waitingRequests.empty() && activeRequests.empty()) {
        return false;
    }
    std::vector<Request*> notYet;
    for (size_t i = 0; i < nonSubmittedRequests.size(); ++i) {
        Request* request = nonSubmittedRequests[i];
        if (request->getAdmissionCycle() > cycle) {
            notYet.push_back(request);
        } else if (canEverFit(*request)) {
            waitingRequests.push_back(request);
        } else {
            std::cout << "rejected " << request->getId() << " cycle=" << cycle << '\n';
            delete request;
        }
    }
    nonSubmittedRequests.swap(notYet);
    admitWaiting();
    ++cycle;
    return true;
}

bool Server::canEverFit(const Request& request) {
    for (size_t g = 0; g < gpus.size(); ++g) {
        if (gpus[g].getBlocksForMemory(request.getMaxNeededMemory()) <= gpus[g].getBlocksNum()) {
            return true;
        }
    }
    return false;
}

void Server::admitWaiting() {
    if (waitingRequests.empty()) return;
    std::vector<Request*> eligible(waitingRequests.begin(), waitingRequests.end());
    //
    // TODO: POLICY
    //
    std::vector<bool> admitted(waitingRequests.size(), false);
    for (size_t i = 0; i < eligible.size(); ++i) {
        //
        // TODO: Write funciton to trys to admit waiting request 
        //
    }
    std::vector<Request*> stillWaiting;
    for (size_t i = 0; i < waitingRequests.size(); ++i) {
        if (!admitted[i]) {
            stillWaiting.push_back(waitingRequests[i]);
        }
    }
    waitingRequests.swap(stillWaiting);
}