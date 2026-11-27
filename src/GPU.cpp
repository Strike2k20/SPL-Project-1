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

std::size_t GPU::getBlocksNum() {
    return blocksNum;
}


