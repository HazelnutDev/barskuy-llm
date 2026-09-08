#pragma once

#include <string>
#include <random>

namespace barskuy::core::utils {

std::string random_id(size_t length = 16);
std::string sha256(const std::string& input);

} // namespace barskuy::core::utils