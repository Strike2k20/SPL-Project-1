#include "GPU.h"
#include "MemoryBlock.h"
#include "Request.h"
#include <algorithm>
#include <stdexcept>
#include <utility>

GPU::SlotInfo::SlotInfo(): hasKey(false), modelID(), key() {}

GPU::GPU(std::size_t blocksNum, std::size_t blockSize):
    blocksNum(blocksNum), blockSize(blockSize), memory(blocksNum, nullptr), info(blocksNum),
    totalReserved(0), cachedHits(0), evictions(0), reservations() {}

// Destructor
GPU::~GPU() {
    clear();
}

// Copies every block of other (keeping its concrete type) and restores the reference counts
void GPU::cloneMemoryFrom(const GPU& other) {
    for (std::size_t i = 0; i < other.memory.size(); ++i) {
        if (other.memory[i] != nullptr) {
            memory[i] = other.memory[i]->clone();
            for (int k = 0; k < other.memory[i]->getRefCount(); ++k) {
                memory[i]->acquire();
            }
        }
    }
}

// Copy Constructor
GPU::GPU(const GPU& other):
    blocksNum(other.blocksNum), blockSize(other.blockSize), memory(other.memory.size(), nullptr),
    info(other.info), totalReserved(other.totalReserved), cachedHits(other.cachedHits),
    evictions(other.evictions), reservations(other.reservations) {
    cloneMemoryFrom(other);
}

// Copy Assignment Operator
GPU& GPU::operator=(const GPU& other) {
    if (this != &other) {
        clear();
        blocksNum = other.blocksNum;
        blockSize = other.blockSize;
        memory.assign(other.memory.size(), nullptr);
        info = other.info;
        totalReserved = other.totalReserved;
        cachedHits = other.cachedHits;
        evictions = other.evictions;
        reservations = other.reservations;
        cloneMemoryFrom(other);
    }
    return *this;
}

// Move Constructor
GPU::GPU(GPU&& other) noexcept:
    blocksNum(other.blocksNum), blockSize(other.blockSize), memory(std::move(other.memory)),
    info(std::move(other.info)), totalReserved(other.totalReserved), cachedHits(other.cachedHits),
    evictions(other.evictions), reservations(std::move(other.reservations)) {
    other.memory.clear();
    other.info.clear();
    other.reservations.clear();
    other.blocksNum = 0;
    other.blockSize = 0;
    other.totalReserved = 0;
    other.cachedHits = 0;
    other.evictions = 0;
}

// Move Assignment Operator
GPU& GPU::operator=(GPU&& other) noexcept {
    if (this != &other) {
        clear();
        blocksNum = other.blocksNum;
        blockSize = other.blockSize;
        memory = std::move(other.memory);
        info = std::move(other.info);
        totalReserved = other.totalReserved;
        cachedHits = other.cachedHits;
        evictions = other.evictions;
        reservations = std::move(other.reservations);

        other.memory.clear();
        other.info.clear();
        other.reservations.clear();
        other.blocksNum = 0;
        other.blockSize = 0;
        other.totalReserved = 0;
        other.cachedHits = 0;
        other.evictions = 0;
    }
    return *this;
}

void GPU::clear() {
    for (std::size_t i = 0; i < memory.size(); ++i) {
        delete memory[i];
        memory[i] = nullptr;
    }
}

std::size_t GPU::getBlocksNum() const {
    return blocksNum;
}

std::size_t GPU::getBlockSize() const {
    return blockSize;
}

// Deletes the block of the slot and empties it (also forgets its prefix key)
void GPU::destroySlot(std::size_t slot) {
    delete memory[slot];
    memory[slot] = nullptr;
    info[slot] = SlotInfo();
}

void GPU::addCacheHits(std::size_t n) {
    cachedHits += n;
}

std::size_t GPU::getCacheHits() const {
    return cachedHits;
}

std::size_t GPU::getEvictions() const {
    return evictions;
}

std::size_t GPU::availableSlots() const {
    std::size_t active = 0;
    for (std::size_t i = 0; i < memory.size(); ++i) {
        if (memory[i] != nullptr && memory[i]->getRefCount() > 0) {
            ++active;
        }
    }
    if (active + totalReserved >= blocksNum) {
        return 0;
    }
    return blocksNum - active - totalReserved;
}

std::size_t GPU::getBlocksForMemory(std::size_t memoryInts) const {
    if (blockSize == 0) {
        return 0;
    }
    return (memoryInts + blockSize - 1) / blockSize;
}

void GPU::reserve(std::size_t n, const Request& request) {
    if (n == 0) {
        return;
    }
    reservations[request.getId()] += n;
    totalReserved += n;
}

void GPU::unreserve(std::size_t n, const Request& request) {
    std::map<int, std::size_t>::iterator it = reservations.find(request.getId());
    if (n == 0 || it == reservations.end()) {
        return;
    }
    if (n > it->second) {
        n = it->second;
    }
    it->second -= n;
    totalReserved -= n;
    if (it->second == 0) {
        reservations.erase(it);
    }
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
    memory[slot]->acquire();
}

void GPU::release(std::size_t slot) {
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
    for (std::size_t i = 0; i < memory.size(); ++i) {
        if (i != block && memory[i] != nullptr && info[i].hasKey && info[i].modelID == modelId
            && info[i].key == prefixTokens) {
            return;
        }
    }
    info[block].hasKey = true;
    info[block].modelID = modelId;
    info[block].key = prefixTokens;
}

int GPU::readFromBlock(std::size_t blockId, std::size_t offset) const {
    return memory[blockId]->readFromBlock(offset);
}

void GPU::writeToBlock(std::size_t blockId, std::size_t offset, int value) {
    memory[blockId]->writeToBlock(offset, value);
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

bool GPU::isIdle(std::size_t slot) const {
    return slot < memory.size() && memory[slot] != nullptr && memory[slot]->getRefCount() == 0;
}