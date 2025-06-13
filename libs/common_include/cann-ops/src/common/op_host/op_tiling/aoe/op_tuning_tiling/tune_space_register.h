/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef TUNE_SPACE_REGISTER_H
#define TUNE_SPACE_REGISTER_H

#include <iostream>
#include <memory>
#include <map>
#include <list>
#include <algorithm>
#include "tune_space.h"
#include "common_utils.h"

namespace OpTuneSpace {
using CreateFunction = std::shared_ptr<TuneSpace>(*)();
// define tune space manager class
class TuneSpaceManager {
public:
    // singleton factory
    static TuneSpaceManager* GetInstance()
    {
        static TuneSpaceManager instance;
        return &instance;
    }

    bool Regist(std::string typeName, CreateFunction func)
    {
        if (!func) {
            return false;
        }
        std::map<std::string, CreateFunction>::const_iterator iter = createMap.find(typeName);
        if (iter == createMap.end()) {
            createMap.insert(make_pair(typeName, func));
        }
        return true;
    }

    std::shared_ptr<TuneSpace> CreateObject(const std::string& typeName)
    {
        if (typeName.empty()){
            return nullptr;
        }
        std::map<std::string, CreateFunction>::const_iterator iter = createMap.find(typeName);
        if (iter == createMap.end()) {
            return nullptr;
        }
        return iter->second();
    }

private:
    std::map<std::string, CreateFunction> createMap;
    TuneSpaceManager() {}
    ~TuneSpaceManager() {}
    TuneSpaceManager(const TuneSpaceManager&) = delete;
    TuneSpaceManager& operator=(const TuneSpaceManager&) = delete;
}; // TuneSpaceManager

// Register Tool
class RegisterClassAction {
public:
    RegisterClassAction(const std::string& className, CreateFunction func)
    {
        TuneSpaceManager::GetInstance()->Regist(className, func);
    }
    ~RegisterClassAction() {}
};

#define TUNE_SPACE_REGISTER(type, clazz) \
    static std::shared_ptr<TuneSpace> Creator_##type##_Class() \
    { \
        std::shared_ptr<clazz> ptr = std::make_shared<clazz>(); \
        return std::shared_ptr<TuneSpace>(ptr); \
    }   \
    RegisterClassAction g_##type##_Class_Creator(#type, Creator_##type##_Class)
}   // namespace OpTuneSpace
#endif