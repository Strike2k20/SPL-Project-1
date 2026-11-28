#ifndef GPU_H_
#define GPU_H_

#include <string>
#include <cstddef>
#include <vector>
#include <map>
#include "Request.h"

struct PrefixMatch {
    std::size_t matchedTokens;
    std::vector<std::size_t> blockIds;
};

class MemoryBlock;

class GPU{
public:
    GPU(std::size_t blocksNum, std::size_t blockSize);
    ~GPU();
    void clear();
    GPU(const GPU& other);
    GPU& operator=(const GPU& other);
    // Copy Assignment Operator, Constructor, Destructor
    GPU(GPU&& other)noexcept;
    GPU& operator=(GPU&& other);
    // Move Assignment Operator, Constructor
    
    std::size_t availableSlots() const;
    std::size_t getBlocksForMemory(std::size_t memory) const;
    void reserve(std::size_t n, const Request& request);
    void unreserve(std::size_t n, const Request& request);
    std::size_t allocateBlock(const Request& request);
    void addMemoryBlock(const GenerativeRequest& request, std::size_t slot);
    void addMemoryBlock(const EmbeddingRequest& request, std::size_t slot);
    void acquire(std::size_t slot);
    void release(std::size_t slot);
    void copyMemoryFromOtherGPU(const GPU& other,const std::vector<std::size_t>& blocks);
    void assignPrefixToBlock(std::size_t block, const std::vector<std::string>& prefixTokens, const std::string& modelId);
    int readFromBlock(std::size_t blockId, std::size_t offset) const;
    void writeToBlock(std::size_t blockId, std::size_t offset, int value);
    PrefixMatch findLongestCachedPrefix(const std::string& modelId, std::size_t embeddingSize,
                                        const std::vector<std::string>& promptTokens) const;

    std::size_t getBlocksNum() const;
    std::size_t getBlockSize() const;
    bool isIdle(std::size_t slot) const;          
    void acquireCached(std::size_t slot);         
    std::size_t getCacheHits() const;
    std::size_t getEvictions() const;
    void addCacheHits(std::size_t n);
private:
    // A block does not know which model or prefix it belongs to: this information is kept by the GPU
    // for each slot
    struct SlotInfo {
        bool hasKey;
        std::string modelID;
        std::vector<std::string> key;
        SlotInfo();
    };

    std::size_t blocksNum;
    std::size_t blockSize;
    std::vector<MemoryBlock*> memory;
    std::vector<SlotInfo> info;
    std::size_t totalReserved;
    std::size_t cachedHits;
    std::size_t evictions;
    std::map<int,std::size_t> reservations;

    void destroySlot(std::size_t slot);
};

#endif