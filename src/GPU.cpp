#include "GPU.h"
#include "MemoryBlock.h"

GPU::GPU(std::size_t blocksNum, std::size_t blockSize): 
    blocksNum(blocksNum), blockSize(blockSize), evictions(0), cachedHits(0),
    memory(blocksNum, nullptr), info(blocksNum), totalReserved(0), reservations() {}

//Deletion 
GPU::~GPU() {
    clear();
}

// Copy Constructor
GPU::GPU(const GPU& other): 
    blocksNum(other.blocksNum), blockSize(other.blockSize), evictions(other.evictions), cachedHits(other.cachedHits),
    memory(other.memory.size(), nullptr), info(other.info), totalReserved(other.totalReserved), reservations(other.reservations) {
        for (std::size_t i = 0; i < other.memory.size(); ++i) {
            if(other.memory[i] != nullptr) {
                memory[i] = other.memory[i]->clone();
                for (std::size_t k = 0; k < other.memory[i]->getRefCount(); ++k) {
                    memory[i]->acquire();
                }
            }
        }
}

//Copy Assignment Operator
GPU& GPU::operator=(const GPU& other) {
    if(this == &other) {
        return *this;
    } else {
        clear();
        blocksNum = other.blocksNum;
        blockSize = other.blockSize;
        evictions = other.evictions;
        cachedHits = other.cachedHits;
        info = other.info;
        totalReserved = other.totalReserved;
        reservations = other.reservations;

        for (std::size_t i = 0; i < other.memory.size(); ++i) {
            if(other.memory[i] != nullptr) {
                memory[i] = other.memory[i]->clone();
                for (std::size_t k = 0; k < other.memory[i]->getRefCount(); ++k) {
                    memory[i]->acquire();
                }
            }
        }
        return *this;
    }
}

// move constructor
GPU::GPU(GPU&& other)noexcept:
    blocksNum(other.blocksNum), blockSize(other.blockSize), evictions(other.evictions), cachedHits(other.cachedHits),
    memory(std::move(other.memory)), info(std::move(other.info)), totalReserved(other.totalReserved), reservations(std::move(other.reservations)) {

    other.blocksNum = 0;
    other.blockSize = 0;
    other.evictions = 0;
    other.cachedHits = 0;
    other.totalReserved = 0;
}

// move assignment
GPU& GPU::operator=(GPU&& other)noexcept {
    if(this == &other){
        return *this;
    } else {
        blocksNum = other.blocksNum;
        blockSize = other.blockSize;
        evictions = other.evictions;
        cachedHits = other.cachedHits;
        totalReserved = other.totalReserved;
        memory = std::move(other.memory);
        info = std::move(other.info);
        reservations = std::move(other.reservations);

        other.blocksNum = 0;
        other.blockSize = 0;
        other.evictions = 0;
        other.cachedHits = 0;
        other.totalReserved = 0;

        return *this;
    }
}

void GPU::clear() {
    for (std::size_t i = 0; i < memory.size(); ++i) {
        delete memory[i];
        memory[i] = nullptr;
    }
}

std::size_t GPU::availableSlots() const {
    std::size_t countBlocks = 0;
    for(std::size_t i = 0; i < memory.size(); ++i) {
        if(memory[i]->getRefCount() > 0) {
            countBlocks++;
        }
    }
    return blockSize - countBlocks - totalReserved;
}

std::size_t GPU::allocateBlock(const Request& request) {
    std::map<int, std::size_t>::iterator it = reservations.find(request.getId());
    if (it == reservations.end() || it->second == 0) {
        throw std::logic_error("GPU::allocateBlock: the request has no reserved memory");
    }
    std::size_t slot = memory.size();
    for (std::size_t i = 0; i < memory.size(); ++i) {
        if (memory[i] == nullptr) {
            slot = i;
            break;
        }
    }
    if (slot == memory.size()) {
        for (std::size_t i = 0; i < memory.size(); ++i) {
            if (memory[i]->getRefCount() == 0) {
                slot = i;
                break;
            }
        }
        if (slot == memory.size()) {
            throw std::logic_error("GPU::allocateBlock: no empty or idle slot");
        }
        destroySlot(slot);
        ++evictions;
    }
    request.allocateOneBlock(*this, slot);
    // consume one block of the reservation
    --it->second;
    --totalReserved;
    if (it->second == 0) {
        reservations.erase(it);
    }
    return slot;
}

void GPU::addMemoryBlock(const GenerativeRequest&, std::size_t slot) {
    delete memory[slot];
    memory[slot] = new KVBlock(blockSize);
    info[slot] = SlotInfo();
}

void GPU::addMemoryBlock(const EmbeddingRequest&, std::size_t slot) {
    delete memory[slot];
    memory[slot] = new EmbeddingWorkspace(blockSize);
    info[slot] = SlotInfo();
}

void GPU::acquire(std::size_t slot) {
    checkSlot(slot);
    memory[slot]->acquire();
}

void GPU::release(std::size_t slot) {
    checkSlot(slot);
    IdleAction action = memory[slot]->release();
    if (memory[slot]->getRefCount() > 0) {
        return;
    }
    if (action == IdleAction::Destroy || !info[slot].hasKey) {
        destroySlot(slot);
    }
}

void GPU::copyMemoryFromOtherGPU(const GPU& other, const std::vector<std::size_t>& blocks) {
    if (this == &other) {
        return;
    }
    // slots that must not be evicted: the blocks of the chain that this GPU already has
    std::vector<bool> protectedSlot(memory.size(), false);
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        const SlotInfo& src = other.info[blocks[b]];
        for (std::size_t i = 0; i < memory.size(); ++i) {
            if (memory[i] != nullptr && info[i].hasKey && info[i].modelID == src.modelID && info[i].key == src.key) {
                protectedSlot[i] = true;
                break;
            }
        }
    }
    for (std::size_t b = 0; b < blocks.size(); ++b) {
        other.checkSlot(blocks[b]);
        const SlotInfo& src = other.info[blocks[b]];
        if (!src.hasKey) {
            continue;
        }
        bool exists = false;
        for (std::size_t i = 0; i < memory.size(); ++i) {
            if (memory[i] != nullptr && info[i].hasKey && info[i].modelID == src.modelID && info[i].key == src.key) {
                exists = true;
                break;
            }
        }
        if (exists) {
            continue;
        }
        std::size_t slot = memory.size();
        for (std::size_t i = 0; i < memory.size(); ++i) {
            if (memory[i] == nullptr) {
                slot = i;
                break;
            }
        }
        if (slot == memory.size()) {
            for (std::size_t i = 0; i < memory.size(); ++i) {
                if (!protectedSlot[i] && memory[i]->getRefCount() == 0) {
                    slot = i;
                    break;
                }
            }
            if (slot == memory.size()) {
                throw std::logic_error("GPU::copyMemoryFromOtherGPU: no free slot to copy into");
            }
            destroySlot(slot);
            ++evictions;
        }
        memory[slot] = other.memory[blocks[b]]->clone();
        info[slot] = src;
        protectedSlot[slot] = true;
    }
}

void GPU::assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId) {
    checkSlot(block);
    for (std::size_t i = 0; i < memory.size(); ++i) {
        if (i != block && memory[i] != nullptr && info[i].hasKey && info[i].modelID == modelId
            && info[i].key == prefixTokens) {
            // somebody already caches this prefix: this block is not cached
            return;
        }
    }
    info[block].hasKey = true;
    info[block].modelID = modelId;
    info[block].key = prefixTokens;
}

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const {
    checkSlot(blockId);
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value) {
    checkSlot(blockId);
    memory[blockId]->writeToBlock(offset, value);
}
std::size_t GPU::getBlocksForMemory(std::size_t memory) const {
    int blocksForMemory = memory / blockSize;
    if(memory % blockSize != 0) {
        return blocksForMemory + 1;
    }
    return blocksForMemory;
}

void GPU::reserve(std::size_t n, const Request& request) {
    if(n > 0){
        reservations[request.getId()] += n;
        totalReserved += n;
    }
}

void GPU::unreserve(std::size_t n, const Request& request) {
    if(n == 0 || reservations.count(request.getId()) == 0) {
        return;
    }
    std::size_t& reserved = reservations[request.getId()];
    std::size_t amount = std::min(n, reserved);
    reserved -= amount;
    totalReserved -= amount;
    if (reserved == 0) {
        reservations.erase(request.getId());
    }
}

PrefixMatch GPU::findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                         const std::vector<std::string>& promptTokens) const {
    PrefixMatch result;
    result.matchedTokens = 0;
    if (embeddingSize == 0 || embeddingSize > blockSize) {
        return result;
    }
    const std::size_t t = blockSize / embeddingSize;
    for (std::size_t len = t; len <= promptTokens.size(); len += t) {
        std::size_t found = memory.size();
        for (std::size_t i = 0; i < memory.size(); ++i) {
            if (memory[i] != nullptr && info[i].hasKey && info[i].key.size() == len
                && info[i].modelID == modelId
                && std::equal(info[i].key.begin(), info[i].key.end(), promptTokens.begin())) {
                found = i;
                break;
            }
        }
        if (found == memory.size()) {
            break;
        }
        result.blockIds.push_back(found);
        result.matchedTokens = len;
    }
    return result;
}


std::size_t GPU::getBlocksNum() const {
    return blocksNum;
}

std::size_t GPU::getBlockSize() const {
    return blockSize;
}

bool GPU::isIdle(std::size_t slot) const {
    return slot < memory.size() && memory[slot] != nullptr && memory[slot]->getRefCount() == 0;
}

void GPU::acquireCached(std::size_t slot) {
    memory[slot]->acquire();
    ++cachedHits;
}

std::size_t GPU::getCacheHits() const {
    return cachedHits;
}

std::size_t GPU::getEvictions() const {
    return evictions;
}

void GPU::addCacheHits(std::size_t n) {
    cachedHits += n;
}

