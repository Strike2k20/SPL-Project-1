#include "GPU.h"
#include "MemoryBlock.h"

GPU::GPU(std::size_t blocksNum, std::size_t blockSize): 
    blocksNum(blocksNum), blockSize(blockSize), evictions(0), cachedHits(0),
    memory(blocksNum, nullptr), info(blocksNum), totalReserved(0) {}

GPU::~GPU() {
    clear();
}

GPU::GPU(const GPU& other): 
    blocksNum(other.blocksNum), blockSize(other.blockSize), evictions(other.evictions), cachedHits(other.cachedHits),
    memory(other.memory.size(), nullptr), info(other.info), totalReserved(other.totalReserved) {
        for (std::size_t i = 0; i < other.memory.size(); ++i) {
            if(other.memory[i] != nullptr) {
                memory[i] = other.memory[i]->clone();
                for (std::size_t k = 0; k < other.memory[i]->getRefCount(); ++k) {
                    memory[i]->acquire();
                }
            }
        }
}

GPU& GPU::operator=(const GPU& other) {
    if(this != &other) {
        clear();
        

    }
}

void GPU::clear() {
    for (std::size_t i = memory.size() - 1; i >= 0; i--) {
        delete memory[i];
        memory[i] = nullptr;
    }
}

std::size_t GPU::getBlocksNum() {
    return blocksNum;
}

