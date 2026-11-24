#include "SchedulingPolicy.h"
#include <vector>

// Sorts the indices by key
// Descending when "descending = True"
// Ties broekn by the lower request id
static std::vector<int> orderByKey(const std::vector<const Request*>& eligible,
     const std::vector<std::size_t>& key, bool descending) {

    std::vector<int> order(eligible.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }

    for (std::size_t i = 1; i < order.size(); ++i) {
        int current = order[i];
        std::size_t j = i;
        while (j > 0) {
            int prev = order[j - 1];
            bool shouldComeBefore = false;
            if (key[current] != key[prev]) {
                if (descending) {
                    shouldComeBefore = key[current] > key[prev];
                } else {
                    shouldComeBefore = key[current] < key[prev];
                }
            } else {
                shouldComeBefore = eligible[current]->getId() < eligible[prev]->getId();
            }
            if (shouldComeBefore) {
                order[j] = prev;
                --j;
            }
        }
        order[j] = current;
    }
    return order;
}

