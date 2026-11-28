#include "SchedulingPolicy.h"
#include <vector>
#include <algorithm>

// Defines how std::sort compares two indices
struct KeyOrderComparator {
    const std::vector<const Request*>& eligible;
    const std::vector<std::size_t>& key;
    bool descending;

    bool operator()(int a, int b) const {
        if (key[a] != key[b]) {
            if (descending) {
                return key[a] > key[b];
            } else {
                return key[a] < key[b];
            }
        }
        return eligible[a]->getId() < eligible[b]->getId();
    }
};

// Sorts the indices by key
// Descending when "descending = True"
// Ties broekn by the lower request id
std::vector<int> SchedulingPolicy::orderByKey(const std::vector<const Request*>& eligible,
     const std::vector<std::size_t>& key, bool descending) {

    std::vector<int> order(eligible.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    KeyOrderComparator comparator{eligible, key, descending};
    std::sort(order.begin(), order.end(), comparator);
    return order;
}

std::vector<int> ArrivalOrderPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const {
    std::vector<std::size_t> key(eligible.size());
    for (std::size_t i = 0; i <eligible.size(); ++i) {
        key[i] = eligible[i]->getAdmissionCycle();
    }
    return orderByKey(eligible, key, false);
}

SchedulingPolicy* ArrivalOrderPolicy::clone() const {
    return new ArrivalOrderPolicy(*this);
}

std::vector<int> ShortestJobPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const {
    std::vector<std::size_t> key(eligible.size());
    for (std::size_t i = 0; i < eligible.size(); ++i) {
        if (gpus.empty()) {
            key[i] = eligible[i]->getMaxNeededMemory();
        } else {
            key[i] = gpus[0].getBlocksForMemory(eligible[i]->getMaxNeededMemory());
        }
    }
    return orderByKey(eligible, key, false);
}

SchedulingPolicy* ShortestJobPolicy::clone() const {
    return new ShortestJobPolicy(*this);
}

std::vector<int> LongestPrefixPolicy::choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const {
    std::vector<std::size_t> key(eligible.size(), 0);
    for (std::size_t i = 0; i < eligible.size(); ++i) {
        for (std::size_t g = 0; g < gpus.size(); ++g) {
            key[i] = std::max(key[i], eligible[i]->findCachedPrefix(gpus[i]).matchedTokens);
        }
    }
    return orderByKey(eligible, key, true);
}

SchedulingPolicy* LongestPrefixPolicy::clone() const {
    return new LongestPrefixPolicy(*this);
}


