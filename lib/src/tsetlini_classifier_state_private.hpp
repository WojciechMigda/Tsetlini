#pragma once

namespace Tsetlini
{


struct ClassifierState;

void initialize_state(ClassifierState & state);
void reset_state_cache(ClassifierState & state);

struct RegressorState;

void initialize_state(RegressorState & state);
void reset_state_cache(RegressorState & state);


} // namespace Tsetlini
