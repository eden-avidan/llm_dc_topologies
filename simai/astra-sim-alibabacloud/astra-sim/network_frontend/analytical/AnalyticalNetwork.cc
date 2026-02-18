/* 
*Copyright (c) 2024, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/

#include"AnalyticalNetwork.h"
#include"AnaSim.h"
#include <fstream>
#include <iostream>

extern std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;
extern map<std::pair<int, std::pair<int, int>>, struct task1> expeRecvHash;
extern map<std::pair<int, std::pair<int, int>>, int> recvHash;
extern map<std::pair<int, std::pair<int, int>>, struct task1> sentHash;
extern map<std::pair<int, int>, int64_t> nodeHash;
extern int local_rank;

AnalyticalNetWork::AnalyticalNetWork(int _local_rank)
    : AstraNetworkAPI(_local_rank) {
  this->npu_offset = 0;
}

AnalyticalNetWork::~AnalyticalNetWork() {}

AstraSim::timespec_t AnalyticalNetWork::sim_get_time() {
  AstraSim::timespec_t timeSpec;
  timeSpec.time_val = AnaSim::Now();
  return timeSpec;
}

void AnalyticalNetWork::sim_schedule(
    AstraSim::timespec_t delta,
    void (*fun_ptr)(void* fun_arg),
    void* fun_arg) {
  AnaSim::Schedule(delta.time_val, fun_ptr, fun_arg);
  return;
}

int AnalyticalNetWork::sim_send(
    void* buffer,
    uint64_t count,
    int type,
    int dst,
    int tag,
    AstraSim::sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {

  // Track data transfer for transport matrix
  if (nodeHash.find(make_pair(rank, 0)) == nodeHash.end()) {
    nodeHash[make_pair(rank, 0)] = count;
  } else {
    nodeHash[make_pair(rank, 0)] += count;
  }

  // Call the message handler immediately (analytical mode)
  if (msg_handler != nullptr) {
    msg_handler(fun_arg);
  }

  return 0;
}

int AnalyticalNetWork::sim_recv(
    void* buffer,
    uint64_t count,
    int type,
    int src,
    int tag,
    AstraSim::sim_request* request,
    void (*msg_handler)(void* fun_arg),
    void* fun_arg) {

  // Track data received for transport matrix
  if (nodeHash.find(make_pair(rank, 1)) == nodeHash.end()) {
    nodeHash[make_pair(rank, 1)] = count;
  } else {
    nodeHash[make_pair(rank, 1)] += count;
  }

  // Call the message handler immediately (analytical mode)
  if (msg_handler != nullptr) {
    msg_handler(fun_arg);
  }

  return 0;
}

// Function to print transport matrix for analytical backend
void print_transport_matrix_analytical() {
  std::ofstream csv_file("transport_matrix_Nitay.csv");
  if (csv_file.is_open()) {
    csv_file << "Node,Sent,Received" << std::endl;
    
    std::map<int, std::pair<int64_t, int64_t>> node_data;
    
    // Collect sent and received data for each node
    for (auto& entry : nodeHash) {
      int node_id = entry.first.first;
      int direction = entry.first.second; // 0 = sent, 1 = received
      int64_t amount = entry.second;
      
      if (direction == 0) {
        node_data[node_id].first = amount; // sent
      } else {
        node_data[node_id].second = amount; // received
      }
    }
    
    // Write data to CSV
    for (auto& entry : node_data) {
      csv_file << entry.first << "," << entry.second.first << "," << entry.second.second << std::endl;
    }
    
    csv_file.close();
    std::cout << "Transport matrix saved to transport_matrix_Nitay.csv" << std::endl;
  } else {
    std::cerr << "Failed to open transport_matrix_Nitay.csv for writing." << std::endl;
  }
}