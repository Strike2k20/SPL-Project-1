#include "Request.h"
#include "Model.h"

Request::Request(int id, const Model& model, int admissionCycle, std::string prompt) 
    : id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)) {}



