#include "MemoryBlock.h"
#include <stdexcept>

MemoryBlock::MemoryBlock(std::size_t size) : size(size), refCount(0), data(new int[size]()) {}

MemoryBlock::~MemoryBlock() {
    delete[] data;
}

MemoryBlock::MemoryBlock(const MemoryBlock& other) : size(other.size), refCount(0), data(new int[other.size]) {
    for (std::size_t i = 0; i < size; ++i) {
        data[i] = other.data[i];
    }
}

MemoryBlock& MemoryBlock::operator=(const MemoryBlock& other) {
    if (this != &other) {
        int* newData = new int[other.size];

        for (std::size_t i = 0; i < other.size; i++) {
            newData[i] = other.data[i];
        }

        delete[] data;
        data = newData;
        size = other.size;
        refCount = 0;
    }

    return *this;
}

MemoryBlock::MemoryBlock(MemoryBlock&& other) : size(other.size), refCount(other.refCount), data(other.data) {
    other.size = 0;
    other.refCount = 0;
    other.data = nullptr;
}

MemoryBlock& MemoryBlock::operator=(MemoryBlock&& other) {
    if (this != &other) {
        delete[] data;
        size = other.size;
        refCount = other.refCount;
        data = other.data;
        other.size = 0;
        other.refCount = 0;
        other.data = nullptr;
    } 
}

void MemoryBlock::writeToBlock(std::size_t offset, int value) {
    if (offset >= size) {
        throw std::out_of_range("MemoryBlock::writeToBlock: offset out of range");
    }
    data[offset] = value;
}

int MemoryBlock::readFromBlock(std::size_t offset) const {
    if (offset >= size) {
        throw std::out_of_range("MemoryBlock::readToBlock: offset out of range");
    }
    return data[offset];
}

void MemoryBlock::acquire() {
    ++refCount;
}

IdleAction MemoryBlock::release() {
    if (refCount > 0) {
        --refCount;
    }
    if (refCount == 0) {
        return onLastRelease();
    }
    return IdleAction::KeepCached;
}

IdleAction KVBlock::onLastRelease() const {
    return IdleAction::KeepCached;
}

IdleAction EmbeddingWorkspace::onLastRelease() const {
    return IdleAction::Destroy;
}

MemoryBlock* KVBlock::clone() const {
    return new KVBlock(*this);
}

MemoryBlock* EmbeddingWorkspace::clone() const {
    return new EmbeddingWorkspace(*this);
}

KVBlock::KVBlock(size_t size) : MemoryBlock(size) {}

EmbeddingWorkspace::EmbeddingWorkspace(size_t size) : MemoryBlock(size) {}