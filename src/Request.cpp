#include "Request.h"
#include "Model.h"



// --------------- Request ---------------

Request::Request(int id, const Model& model, int admissionCycle, std::string prompt)
    : id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)),
      memoryBlocks(), reservedBlocks(0) {}

Request::Request(const Request& other, const Model& newModel)
    : id(other.id), model(newModel), admissionCycle(other.admissionCycle), tokens(other.tokens),
      memoryBlocks(other.memoryBlocks), reservedBlocks(other.reservedBlocks) {}

Request::~Request() {}

int Request::getId() const {
    return id;
}

const Model& Request::getModel() const {
    return model;
}

int Request::getAdmissionCycle() const {
    return admissionCycle;
}

void Request::deallocateMemory(GPU& gpu) {
    const std::size_t unused = getNeedToUnreserve();
    if (unused > 0) {
        gpu.unreserve(unused, *this);
    }
    reservedBlocks = 0;
    for (std::size_t i = 0; i < memoryBlocks.size(); ++i) {
        gpu.release(memoryBlocks[i]);
    }
    memoryBlocks.clear();
}

PrefixMatch Request::findCachedPrefix(const GPU& gpu) const {
    return gpu.findLongestCachedPrefix(model.getId(), model.getEmbeddingSize(), getCacheableTokens());
}

// --------------- GenerativeRequest ---------------

GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt,
                                     std::string scriptedOutput, std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput),
      maxOutputTokens(maxOutputTokens), outputTokens(model.tokenizer(scriptedOutput)), computed(0) {
    if (outputTokens.size() > maxOutputTokens) {
        outputTokens.resize(maxOutputTokens);
    }
}

GenerativeRequest::GenerativeRequest(const GenerativeRequest& other, const Model& newModel)
    : Request(other, newModel), scriptedOutput(other.scriptedOutput), maxOutputTokens(other.maxOutputTokens),
      outputTokens(other.outputTokens), computed(other.computed) {}

Request* GenerativeRequest::clone(const Model& newModel) const {
    return new GenerativeRequest(*this, newModel);
}

std::size_t GenerativeRequest::getNeededMemory(const GPU& gpu) const {
    const std::size_t h = findCachedPrefix(gpu).matchedTokens;
    return model.getEmbeddingSize() * (tokens.size() - h + maxOutputTokens);
}

std::size_t GenerativeRequest::getMaxNeededMemory() const {
    return model.getEmbeddingSize() * (tokens.size() + maxOutputTokens);
}

std::vector<std::string> GenerativeRequest::getCacheableTokens() const {
    return tokens;
}

void GenerativeRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const {
    gpu.addMemoryBlock(*this, slot);     
}

void GenerativeRequest::reserveMemory(GPU& gpu) {
    const PrefixMatch match = findCachedPrefix(gpu);
    const std::size_t needed = gpu.getBlocksForMemory (
    model.getEmbeddingSize() * (tokens.size() - match.matchedTokens + maxOutputTokens));
    for (std::size_t i = 0; i < match.blockIds.size(); ++i) {
        gpu.acquire(match.blockIds[i]);
        memoryBlocks.push_back(match.blockIds[i]);
    }
    gpu.addCacheHits(match.blockIds.size());
    computed = match.matchedTokens;
    gpu.reserve(needed, *this);
    reservedBlocks += needed;
}

bool GenerativeRequest::finished() const {
    return computed >= tokens.size() + outputTokens.size();
}

void GenerativeRequest::act(GPU& gpu) {
    if (finished()) {
        return;
    }

    const std::size_t preBlock = gpu.getBlockSize() / model.getEmbeddingSize();
    const std::size_t blockIndex = computed / preBlock;
    const std::size_t offset = (computed % preBlock) * model.getEmbeddingSize();
    if (blockIndex >= memoryBlocks.size()) {
        const std::size_t slot = gpu.allocateBlock(*this);
        gpu.acquire(slot);
        memoryBlocks.push_back(slot);
        --reservedBlocks;
    }
    const std::size_t slot = memoryBlocks[blockIndex];

    std::vector<std::string> prefix;
    prefix.reserve(computed + 1);
    for (std::size_t i = 0; i <= computed; ++i) {
        if (i < tokens.size()) {
            prefix.push_back(tokens[i]);
        } else {
            prefix.push_back(outputTokens[i - tokens.size()]);
        }
    }
    model.generateTokenEmbedding(prefix, gpu, slot, offset);

    if (computed % preBlock == preBlock - 1) {
        gpu.assignPrefixToBlock(slot, prefix, model.getId());
    }

    ++computed;
}

std::string GenerativeRequest::getOutput() const {
    std::string output;
    const std::size_t generated = computed > tokens.size() ? computed - tokens.size() : 0;
    for (std::size_t i = 0; i < generated && i < outputTokens.size(); ++i) {
        if (i > 0) {
            output += " ";
        }
        output += outputTokens[i];
    }
    return output;
}

std::size_t GenerativeRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}

// --------------- EmbeddingRequest ---------------

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt)
    : Request(id, model, admissionCycle, prompt), processed(0), result(model.getEmbeddingSize(), 0) {}

EmbeddingRequest::EmbeddingRequest(const EmbeddingRequest& other, const Model& newModel)
    : Request(other, newModel), processed(other.processed), result(other.result) {}

Request* EmbeddingRequest::clone(const Model& newModel) const {
    return new EmbeddingRequest(*this, newModel);
}

std::size_t EmbeddingRequest::getNeededMemory(const GPU&) const {
    return 2 * model.getEmbeddingSize();
}

std::size_t EmbeddingRequest::getMaxNeededMemory() const {
    return 2 * model.getEmbeddingSize();
}

std::vector<std::string> EmbeddingRequest::getCacheableTokens() const {
    return std::vector<std::string>();
}

void EmbeddingRequest::allocateOneBlock(GPU& gpu, std::size_t slot) const {
    gpu.addMemoryBlock(*this, slot);     
}

void EmbeddingRequest::reserveMemory(GPU& gpu) {
    reservedBlocks = gpu.getBlocksForMemory(getNeededMemory(gpu));
    gpu.reserve(reservedBlocks, *this);
    while (reservedBlocks > 0) {
        memoryBlocks.push_back(gpu.allocateBlock(*this));
        --reservedBlocks;
    }
}

std::size_t EmbeddingRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}

void EmbeddingRequest::act(GPU& gpu) {
    if (processed >= tokens.size()) {
        return;
    }
    const std::size_t e = model.getEmbeddingSize();
    const std::size_t blockSize = gpu.getBlockSize();

    std::vector<std::string> current(1, tokens[processed]);
    model.generateTokenEmbedding(current, gpu, memoryBlocks[e / blockSize], e % blockSize);
    for (std::size_t i = 0; i < e; ++i) {
        const int sum = gpu.readFromBlock(memoryBlocks[i / blockSize], i % blockSize);
        const int value = gpu.readFromBlock(memoryBlocks[(e + i) / blockSize], (e + i) % blockSize);
        gpu.writeToBlock(memoryBlocks[i / blockSize], i % blockSize, sum + value);
    }
    ++processed;

    if (processed == tokens.size()) {        
        for (std::size_t i = 0; i < e; ++i) {
            result[i] = gpu.readFromBlock(memoryBlocks[i / blockSize], i % blockSize) / (tokens.size());
        }
    }
}

bool EmbeddingRequest::finished() const {
    return processed >= tokens.size();
}

std::string EmbeddingRequest::getOutput() const {
    std::string out = "[";
    for (std::size_t j = 0; j < result.size(); j++) {
        if (j > 0) {
            out += ", ";
        }
        out += std::to_string(result[j]);
    }
    out += "]";
    return out;
}

std::size_t EmbeddingRequest::getNeedToUnreserve() const {
    return reservedBlocks;
}