#include "MemoryBlock.h"

MemoryBlock::MemoryBlock(size_t size) {
    this->size = size;
    refCount = 0;
    data = new int[size]();
}