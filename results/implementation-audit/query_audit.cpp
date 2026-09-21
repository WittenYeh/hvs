#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
#include <omp.h>
#include <opencv2/core.hpp>
#ifdef ORIGINAL
#include "sift_1b.cpp"
#undef L
#undef OFF
#undef min_book
#undef cen
#undef nnum
#undef fan
#undef KK
#undef size_n
#include "original_adapter.hpp"
extern "C" time_t __wrap_time(time_t* out){if(out)*out=100;return 100;}
#else
#include "hvs.cpp"
#endif
template<class T> void write_values(std::ofstream& out, const T* p, size_t n) {
    out.write(reinterpret_cast<const char*>(p), n * sizeof(T));
}
std::vector<float> read_vectors(const char* filename, int count, int dimension) {
    std::ifstream input(filename, std::ios::binary);
    std::vector<float> result(size_t(count) * dimension);
    for (int row = 0; row < count; ++row) {
        int stored;
        input.read(reinterpret_cast<char*>(&stored), sizeof(stored));
        if (stored != dimension) throw std::runtime_error("dimension mismatch");
        input.read(reinterpret_cast<char*>(result.data() + size_t(row)*dimension), dimension*sizeof(float));
        if (!input) throw std::runtime_error("truncated vectors");
    }
    return result;
}
#ifndef ORIGINAL
void dump_index(hvs::Index& index) {
    auto& impl = *index.impl_;
    auto& graph = *impl.graph;
    graph.saveIndex("index.bin", nullptr);
    {
        std::ofstream out("index2.bin", std::ios::binary);
        write_values(out, &graph.max_elements_, 1);
        write_values(out, &graph.size_data_per_element_, 1);
        write_values(out, graph.data_level0_memory_, graph.max_elements_*graph.size_data_per_element_);
    }
    {
        std::ofstream out("quantizer.gt", std::ios::binary);
        write_values(out, impl.lengths.data(), impl.levels);
        write_values(out, impl.subdimensions.data(), impl.levels);
        write_values(out, impl.counts.data(), impl.levels);
    }
    {
        std::ofstream out("searching.gt", std::ios::binary);
        const int subdim = impl.subdimensions[0], width = impl.width;
        for (int block = 0; block < impl.lengths[0]; ++block)
            write_values(out, impl.packed_codebooks.data()+size_t(block)*subdim*(width+256), subdim*width);
        for (int level = 0; level < impl.levels; ++level) {
            std::vector<char> flags(impl.n);
            std::copy(impl.flags[level].begin(), impl.flags[level].end(), flags.begin());
            write_values(out, flags.data(), flags.size());
        }
        for (int level = 0; level < impl.levels; ++level)
            write_values(out, impl.merges[level].data(), impl.merges[level].size());
        write_values(out, impl.start_merges.data(), impl.start_merges.size());
        for (int block = 0; block < impl.lengths[0]; ++block)
            write_values(out, impl.packed_codebooks.data()+size_t(block)*subdim*(width+256)+subdim*width, 256*subdim);
        write_values(out, impl.connections.data(), impl.connections.size());
        for (const auto& transition : impl.transitions)
            write_values(out, transition.data(), transition.size());
    }
}

#endif

int main(int argc,char** argv){try{
    if(argc!=8)throw std::runtime_error("mode base query n dim T nq");
    cv::setNumThreads(1);cv::setRNGSeed(100);omp_set_dynamic(0);
    const std::string mode=argv[1];
    const int n=std::stoi(argv[4]),dim=std::stoi(argv[5]),levels=std::stoi(argv[6]),nq=std::stoi(argv[7]);
    const int threads=omp_get_max_threads();
    int actual=0;
    #pragma omp parallel
    {
        #pragma omp single
        actual=omp_get_num_threads();
    }
    if(actual!=threads)throw std::runtime_error("OpenMP did not create requested maximum thread count");
    std::cout<<"AUDIT_THREADS "<<actual<<std::endl;
#ifdef ORIGINAL
    if(mode=="build"){
        sift_test1B(argv[2],argv[3],n,dim,levels,-1,.5f,-1);return 0;
    }
    OriginalIndex index(n,dim,levels);
    using Scratch=OriginalIndex::Scratch;
#else
    auto base=read_vectors(argv[2],n,dim);
    hvs::BuildConfig config;config.T=levels;
    hvs::Index index(base.data(),n,dim,config);
    dump_index(index);
    using Scratch=hvs::Index::QueryScratch;
#endif
    auto queries=read_vectors(argv[3],nq,dim);
    const size_t capacity=std::min(size_t(256),index.max_search_queue());
    std::vector<std::unique_ptr<Scratch>> workers;
    for(int t=0;t<threads;++t)workers.emplace_back(new Scratch(index,capacity));
    std::map<std::string,std::vector<unsigned>> previous;
    size_t total=0,configurations=0;
    for(size_t ef:{256,1,32,8,128,16,64,256})for(size_t k:{1,10,100}){
        if(ef<k||ef>capacity)continue;
        std::vector<unsigned> ids(size_t(nq)*k);
        #pragma omp parallel for schedule(dynamic,1)
        for(int q=0;q<nq;++q)index.search(queries.data()+size_t(q)*dim,k,ef,ids.data()+size_t(q)*k,*workers[omp_get_thread_num()]);
        for(int q=0;q<nq;++q){
            std::vector<unsigned> row(ids.begin()+q*k,ids.begin()+(q+1)*k);
            std::sort(row.begin(),row.end());
            if(row.back()>=unsigned(n)||std::adjacent_find(row.begin(),row.end())!=row.end())throw std::runtime_error("Invalid returned IDs");
        }
        const auto name="ids-k"+std::to_string(k)+"-ef"+std::to_string(ef)+".bin";
        if(previous.count(name)&&previous[name]!=ids)throw std::runtime_error("Scratch reuse changed results");
        previous[name]=ids;
        std::ofstream out(name,std::ios::binary);write_values(out,ids.data(),ids.size());
        total+=ids.size();++configurations;
    }
    std::cout<<"AUDIT_OK configurations="<<configurations<<" ids="<<total<<" threads="<<threads<<std::endl;
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
