#ifndef SCHEDULING_POLICY_H_
#define SCHEDULING_POLICY_H

#include "Request.h"

class SchedulingPolicy {
public:
    virtual ~SchedulingPolicy() = default;
    virtual std::vector<int> choose(const std::vector<const Request*>& eligible, const std::vector<GPU>& gpus) const = 0;
    virtual SchedulingPolicy* clone() const = 0;
    static std::vector<int> orderByKey(const std::vector<const Request*>& eligible, const std::vector<std::size_t>& key, bool descending);
};

class ArrivalOrderPolicy : public SchedulingPolicy { 

};
class ShortestJobPolicy : public SchedulingPolicy { 

};
class LongestPrefixPolicy : public SchedulingPolicy {

};


#endif