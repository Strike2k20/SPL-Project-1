#include <fstream>
#include <stdexcept>
#include "json.hpp"

#include <iostream>
#include "Server.h"
#include "Model.h"
#include "SchedulingPolicy.h"

using namespace std;
using json = nlohmann::json;

Config Server::readConfig(string configPath) const
{
    std::ifstream input(configPath);
    json config;
    input >> config;
    // All the GPUs have the same block size.
    size_t blockSize = config["block_size"].get<size_t>();
    vector<GPUConfig> gpuConfigs;
    for (auto &gpu : config["gpus"])
    {
        size_t memoryBytes = gpu["memory_bytes"].get<size_t>();
        GPUConfig gpuConfig{memoryBytes};
        gpuConfigs.push_back(gpuConfig);
    }
    vector<ModelConfig> modelConfigs;
    for (auto &model : config["models"])
    {
        string id = model["id"].get<string>();
        size_t embeddingSize = model["embedding_size"].get<size_t>();
        ModelConfig modelConfig{id, embeddingSize};
        modelConfigs.push_back(modelConfig);
    }
    vector<RequestConfig> requestConfigs;
    for (auto &request : config["requests"])
    {
        size_t id = request["id"].get<size_t>();
        string typeStr = request["type"].get<string>();
        if (typeStr != "generative" && typeStr != "text_embedder")
        {
            throw invalid_argument("invalid request type");
        }
        RequestType type = typeStr == "generative" ? RequestType::Generative : RequestType::TextEmbedder;
        string modelId = request["model"].get<string>();
        int admissionCycle = request["cycle"].get<int>();
        string prompt = request["prompt"].get<string>();
        string scriptedOutput = "";
        if (request.count("scripted_output") > 0)
        {
            scriptedOutput = request["scripted_output"].get<string>();
        }
        size_t maxOutputTokens = 0;
        if (request.count("max_output_tokens") > 0)
        {
            maxOutputTokens = request["max_output_tokens"].get<size_t>();
        }
        RequestConfig reqConfig{id, type, modelId, admissionCycle, prompt, scriptedOutput, maxOutputTokens};
        requestConfigs.push_back(reqConfig);
    }

    if (config.count("scheduling_policy") == 0)
    {
        throw invalid_argument("missing scheduling policy");
    }
    string policy = config["scheduling_policy"].get<string>();

    return Config{blockSize, gpuConfigs, modelConfigs, requestConfigs, policy};
}

Server::Server(std::string configPath)
    : gpus(), models(), nonSubmittedRequests(), waitingRequests(), activeRequests(), policy(nullptr),
      activeGpu(), cycle(0)
{
    Config config = readConfig(configPath);
    if (config.blockSize == 0)
    {
        throw invalid_argument("block_size must be positive");
    }
    for (size_t i = 0; i < config.gpus.size(); ++i)
    {
        gpus.push_back(GPU(config.gpus[i].memoryBytes / config.blockSize, config.blockSize));
    }
    for (size_t i = 0; i < config.models.size(); ++i)
    {
        models.push_back(new Model(config.models[i].id, config.models[i].embeddingSize));
    }
    for (size_t i = 0; i < config.requests.size(); ++i)
    {
        const RequestConfig &rc = config.requests[i];
        const Model *model = nullptr;
        for (size_t m = 0; m < models.size() && model == nullptr; ++m)
        {
            if (models[m]->getId() == rc.modelId)
            {
                model = models[m];
            }
        }
        if (model == nullptr)
        {
            throw invalid_argument("request refers to an unknown model");
        }
        Request *request = nullptr;
        if (rc.type == RequestType::Generative)
        {
            request = new GenerativeRequest(rc.id, *model, rc.admissionCycle, rc.prompt,
                                            rc.scriptedOutput, rc.maxOutputTokens);
        }
        else
        {
            request = new EmbeddingRequest(rc.id, *model, rc.admissionCycle, rc.prompt);
        }
        nonSubmittedRequests.push_back(request);
    }
    if (config.policy == "arrivalOrder")
    {
        policy = new ArrivalOrderPolicy();
    }
    else if (config.policy == "shortestJob")
    {
        policy = new ShortestJobPolicy();
    }
    else if (config.policy == "longestPrefix")
    {
        policy = new LongestPrefixPolicy();
    }
    else
    {
        throw invalid_argument("unknown scheduling policy");
    }
}

// Deletes what the server owns through raw pointers.
void Server::clear() {
    for (size_t i = 0; i < nonSubmittedRequests.size(); ++i) delete nonSubmittedRequests[i];
    for (size_t i = 0; i < waitingRequests.size(); ++i) delete waitingRequests[i];
    for (size_t i = 0; i < activeRequests.size(); ++i) delete activeRequests[i];
    for (size_t i = 0; i < models.size(); ++i) delete models[i];
    delete policy;
    nonSubmittedRequests.clear();
    waitingRequests.clear();
    activeRequests.clear();
    models.clear();
    activeGpu.clear();
    policy = nullptr;
}

Server::~Server() {
    clear();
}

// Deep copy of modles, requests and policy
void Server::cloneFrom(const Server& other) {
    for (size_t i = 0; i < other.models.size(); ++i) {
        models.push_back(new Model(*other.models[i]));
    }

    const std::vector<Request*>* sources[3] = {
        &other.nonSubmittedRequests,
        &other.waitingRequests,
        &other.activeRequests
    };

    for (int k = 0; k < 3; ++k) {
        for (size_t i = 0; i < sources[k]->size(); ++i) {
            const Request* source = (*sources[k])[i];
            const Model* newModel = nullptr;

            for (size_t m = 0; m < models.size() && newModel == nullptr; ++m) {
                if (models[m]->getId() == source->getModel().getId()) {
                    newModel = models[m];
                }
            }

            Request* clonedRequest = nullptr;
            if (newModel != nullptr) {
                clonedRequest = source->clone(*newModel);
            }

            if (clonedRequest != nullptr) {
                if (k == 0) {
                    nonSubmittedRequests.push_back(clonedRequest);
                } else if (k == 1) {
                    waitingRequests.push_back(clonedRequest);
                } else if (k == 2) {
                    activeRequests.push_back(clonedRequest);
                }
            }
        }
    }

    if (other.policy == nullptr) {
        policy = nullptr;
    } else {
        policy = other.policy->clone();
    }
}

Server::Server(const Server& other)
    : gpus(other.gpus), models(), nonSubmittedRequests(), waitingRequests(), activeRequests(), policy(nullptr),
      activeGpu(other.activeGpu), cycle(other.cycle) {
        cloneFrom(other);
}

Server& Server::operator=(const Server& other) {
    if (this == &other) {
        return *this;
    }
    clear();
    gpus = other.gpus;
    activeGpu = other.activeGpu;
    cycle = other.cycle;
    cloneFrom(other);
    return *this;
}

Server& Server::operator=(Server&& other) noexcept {
    if (this != &other) {
        clear();
        gpus = std::move(other.gpus);
        models = std::move(other.models);
        nonSubmittedRequests = std::move(other.nonSubmittedRequests);
        waitingRequests = std::move(other.waitingRequests);
        activeRequests = std::move(other.activeRequests);
        activeGpu = std::move(other.activeGpu);
        policy = other.policy;
        cycle = other.cycle;
        other.gpus.clear();
        other.models.clear();
        other.nonSubmittedRequests.clear();
        other.waitingRequests.clear();
        other.activeRequests.clear();
        other.activeGpu.clear();
        other.policy = nullptr;
        other.cycle = 0;
    }
    return *this;
}

// The moved-from server owns nothing: no GPUs, models, requests or policy, cycle 0.
Server::Server(Server&& other) noexcept
    : gpus(std::move(other.gpus)), models(std::move(other.models)),
      nonSubmittedRequests(std::move(other.nonSubmittedRequests)),
      waitingRequests(std::move(other.waitingRequests)), activeRequests(std::move(other.activeRequests)),
      policy(other.policy), activeGpu(std::move(other.activeGpu)), cycle(other.cycle) {
    other.gpus.clear();
    other.models.clear();
    other.nonSubmittedRequests.clear();
    other.waitingRequests.clear();
    other.activeRequests.clear();
    other.activeGpu.clear();
    other.policy = nullptr;
    other.cycle = 0;
}

bool Server::step()
{
    if (nonSubmittedRequests.empty() && waitingRequests.empty() && activeRequests.empty())
    {
        return false;
    }
    std::vector<Request *> notYet;
    for (size_t i = 0; i < nonSubmittedRequests.size(); ++i)
    {
        Request *request = nonSubmittedRequests[i];
        if (request->getAdmissionCycle() > cycle)
        {
            notYet.push_back(request);
        }
        else if (canEverFit(*request))
        {
            waitingRequests.push_back(request);
        }
        else
        {
            std::cout << "rejected " << request->getId() << " cycle=" << cycle << '\n';
            delete request;
        }
    }
    nonSubmittedRequests.swap(notYet);
    admitWaiting();
    actActive();
    ++cycle;
    return true;
}

// A request that does not fit on any GPU, even an empty one, is rejected.
bool Server::canEverFit(const Request &request) const
{
    for (size_t g = 0; g < gpus.size(); ++g)
    {
        if (gpus[g].getBlocksForMemory(request.getMaxNeededMemory()) <= gpus[g].getBlocksNum())
        {
            return true;
        }
    }
    return false;
}

void Server::admitWaiting()
{
    if (waitingRequests.empty())
        return;
    std::vector<const Request *> eligible(waitingRequests.begin(), waitingRequests.end());
    const std::vector<int> order = policy->choose(eligible, gpus);
    std::vector<bool> admitted(waitingRequests.size(), false);
    for (size_t i = 0; i < order.size(); ++i)
    {
        // A request that does not fit is skipped; later ones are still examined.
        if (tryAdmit(waitingRequests[order[i]]))
            admitted[order[i]] = true;
    }
    std::vector<Request *> stillWaiting;
    for (size_t i = 0; i < waitingRequests.size(); ++i)
    {
        if (!admitted[i])
            stillWaiting.push_back(waitingRequests[i]);
    }
    waitingRequests.swap(stillWaiting);
}

// Admission (Section 4.1): choose the GPU, copy a longer cached prefix from another GPU if there is one,
// then let the request reserve its memory.
bool Server::tryAdmit(Request *request)
{
    bool found = false;
    size_t best = 0, bestCached = 0, bestNeeded = 0;
    PrefixMatch bestMatch;
    for (size_t g = 0; g < gpus.size(); ++g)
    {
        PrefixMatch match = request->findCachedPrefix(gpus[g]);
        size_t idle = 0; // idle cached blocks take a slot when acquired
        for (size_t i = 0; i < match.blockIds.size(); ++i)
        {
            if (gpus[g].isIdle(match.blockIds[i]))
                ++idle;
        }
        const size_t needed = gpus[g].getBlocksForMemory(request->getNeededMemory(gpus[g])) + idle;
        if (needed > gpus[g].availableSlots())
            continue;
        const size_t cached = match.blockIds.size();
        // Most cached blocks, then fewest needed blocks, then the lower index (strict comparisons).
        if (!found || cached > bestCached || (cached == bestCached && needed < bestNeeded))
        {
            found = true;
            best = g;
            bestCached = cached;
            bestNeeded = needed;
            bestMatch = match;
        }
    }
    if (!found)
        return false;

    // Another GPU with a longer cached prefix: copy its blocks (longest prefix, then lower index).
    size_t source = best;
    size_t sourceTokens = bestMatch.matchedTokens;
    PrefixMatch sourceMatch;
    for (size_t g = 0; g < gpus.size(); ++g)
    {
        if (g == best)
            continue;
        PrefixMatch match = request->findCachedPrefix(gpus[g]);
        if (match.matchedTokens > sourceTokens)
        {
            source = g;
            sourceTokens = match.matchedTokens;
            sourceMatch = match;
        }
    }
    if (source != best)
    {
        gpus[best].copyMemoryFromOtherGPU(gpus[source], sourceMatch.blockIds);
    }

    request->reserveMemory(gpus[best]);
    activeRequests.push_back(request);
    activeGpu.push_back(best);
    return true;
}

void Server::actActive()
{
    std::vector<Request *> remaining;
    std::vector<size_t> remainingGpu;
    for (size_t i = 0; i < activeRequests.size(); ++i)
    {
        Request *request = activeRequests[i];
        GPU &gpu = gpus[activeGpu[i]];
        request->act(gpu);
        if (request->finished())
        {
            std::cout << "finished " << request->getId() << " cycle=" << cycle << " gpu=" << activeGpu[i]
                      << ": " << request->getOutput() << '\n';
            request->deallocateMemory(gpu); // releases blocks and the unused reservation
            delete request;
        }
        else
        {
            remaining.push_back(request);
            remainingGpu.push_back(activeGpu[i]);
        }
    }
    activeRequests.swap(remaining);
    activeGpu.swap(remainingGpu);
}

void Server::serve() {
    while (step()) {
    }
    size_t hits = 0, evictions = 0;
    for (size_t g = 0; g < gpus.size(); ++g) {
        hits += gpus[g].getCacheHits();
        evictions += gpus[g].getEvictions();
    }
    std::cout << "summary: cycles=" << cycle << " cache_hits=" << hits << " evictions=" << evictions << '\n';
}
