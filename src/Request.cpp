#include "Request.h"
#include "Model.h"


// --------------- Request ---------------

Request::Request(int id, const Model& model, int admissionCycle, std::string prompt)
    : id(id), model(model), admissionCycle(admissionCycle), tokens(model.tokenizer(prompt)),
      memoryBlocks(), reservedBlocks(0) {}

Request::~Request() {}

int Request::getId() const {
    return id;
}

const Model& Request::getModel() const {
    return model;
}

int Request::getAdmissionCycle() const {
    return admissionCycle;
}

// --------------- GenerativeRequest ---------------

GenerativeRequest::GenerativeRequest(int id, const Model& model, int admissionCycle, std::string prompt,
                                     std::string scriptedOutput, std::size_t maxOutputTokens)
    : Request(id, model, admissionCycle, prompt), scriptedOutput(scriptedOutput),
      maxOutputTokens(maxOutputTokens), outputTokens(model.tokenizer(scriptedOutput)), computed(0) {
    if (outputTokens.size() > maxOutputTokens) {
        outputTokens.resize(maxOutputTokens);
    }
}

// --------------- EmbeddingRequest ---------------

EmbeddingRequest::EmbeddingRequest(int id, const Model& model, int admissionCycle, std::string prompt)
    : Request(id, model, admissionCycle, prompt), processed(0), result(model.getEmbeddingSize(), 0) {}

bool EmbeddingRequest::finished() const {
    return processed >= tokens.size();
}