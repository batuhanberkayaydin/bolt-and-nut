/**
 * @file jsonConfig.hpp
 * @author Berkay
 * @brief Parameter loader header file for processes
 *
 */

#ifndef CONFIG_H
#define CONFIG_H

#include "iostream"
#include "fstream"
#include "json.hpp"


class configParams
{

public:
    configParams();
    ~configParams();
    void readParams();
    nlohmann::json jsonData_;

    nlohmann::json getParams();
    
private:


};


#endif // CONFIG_H
