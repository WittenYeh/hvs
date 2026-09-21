#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#include <opencv2/core.hpp>

#ifdef ORIGINAL
static double perf_native_build_seconds=0;
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
#else
#include <hvs.hpp>
#endif
extern "C" time_t __wrap_time(time_t* out){if(out)*out=100;return 100;}
using Clock=std::chrono::steady_clock;
static double elapsed(Clock::time_point start){return std::chrono::duration<double>(Clock::now()-start).count();}
static std::vector<float> fvecs(const std::string& path,int dimension,size_t limit=0){
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    if(!input)throw std::runtime_error("Cannot open "+path);
    size_t count=size_t(input.tellg())/(4+4*dimension);if(limit)count=std::min(count,limit);
    std::vector<float> data(count*dimension);input.seekg(0);
    for(size_t i=0;i<count;++i){int d;input.read(reinterpret_cast<char*>(&d),4);
        if(d!=dimension)throw std::runtime_error("Wrong fvecs dimension");
        input.read(reinterpret_cast<char*>(data.data()+i*dimension),dimension*4);if(!input)throw std::runtime_error("Truncated fvecs");}
    return data;
}
static std::vector<unsigned> ivecs(const std::string& path,size_t nq,int& width){
    std::ifstream input(path,std::ios::binary);input.read(reinterpret_cast<char*>(&width),4);
    if(!input||width<100)throw std::runtime_error("Ground truth needs at least 100 neighbors");
    input.seekg(0);std::vector<unsigned> gt(nq*width);
    for(size_t i=0;i<nq;++i){int d;input.read(reinterpret_cast<char*>(&d),4);if(d!=width)throw std::runtime_error("Wrong ivecs width");
        input.read(reinterpret_cast<char*>(gt.data()+i*width),width*4);if(!input)throw std::runtime_error("Truncated ivecs");}
    return gt;
}
int main(int argc,char** argv){try{
    if(argc!=9&&argc!=10)throw std::runtime_error("mode base query truth n dim T build_threads [verify]");
    const std::string mode=argv[1];const int n=std::stoi(argv[5]),dimension=std::stoi(argv[6]),levels=std::stoi(argv[7]),build_threads=std::stoi(argv[8]);
    const bool verify=argc==10;
    cv::setNumThreads(1);cv::setRNGSeed(100);std::srand(100);omp_set_dynamic(0);omp_set_num_threads(build_threads);
    std::cout<<std::setprecision(17)<<std::unitbuf;
#ifdef ORIGINAL
    const char* variant="upstream";
    if(mode=="build"){
        const auto start=Clock::now();
        sift_test1B(argv[2],argv[3],n,dimension,levels,-1,0.5f,-1);
        const double seconds=elapsed(start);
        std::cout<<"PERF {\"kind\":\"build\",\"variant\":\"upstream\",\"build_api_s\":"<<seconds
                 <<",\"native_build_s\":"<<perf_native_build_seconds<<",\"build_threads\":"<<build_threads<<"}\n";
        return 0;
    }
    const auto start=Clock::now();OriginalIndex index(n,dimension,levels);const double reload=elapsed(start);
    std::cout<<"PERF {\"kind\":\"load\",\"variant\":\"upstream\",\"index_load_s\":"<<reload<<"}\n";
    using Scratch=OriginalIndex::Scratch;
#else
    const char* variant="current";
    const auto input_start=Clock::now();auto base=fvecs(argv[2],dimension,n);const double input_seconds=elapsed(input_start);
    if(base.size()!=size_t(n)*dimension)throw std::runtime_error("Wrong base size");
    hvs::BuildConfig config;config.T=levels;
    const auto start=Clock::now();hvs::Index index(base.data(),n,dimension,config);const double build_seconds=elapsed(start);
    std::cout<<"PERF {\"kind\":\"build\",\"variant\":\"current\",\"input_load_s\":"<<input_seconds
             <<",\"build_api_s\":"<<build_seconds<<",\"ready_s\":"<<(input_seconds+build_seconds)<<",\"build_threads\":"<<build_threads<<"}\n";
    using Scratch=hvs::Index::QueryScratch;
#endif
    auto queries=fvecs(argv[3],dimension);const size_t nq=queries.size()/dimension;
    int gt_width=0;auto gt=ivecs(argv[4],nq,gt_width);
    const size_t capacity=std::min(size_t(512),index.max_search_queue());
    const std::vector<size_t> efs=verify?std::vector<size_t>{1,8,16,32,64,128,256}:std::vector<size_t>{1,2,4,8,12,16,24,32,48,64,96,128,192,256,384,512};
    for(int threads:(verify?std::vector<int>{1}:std::vector<int>{1,16})){
        std::vector<std::unique_ptr<Scratch>> scratch;
        for(int t=0;t<threads;++t)scratch.emplace_back(new Scratch(index,capacity));
        std::vector<std::pair<size_t,size_t>> configs;
        for(size_t k:{1,10,100})for(size_t ef:efs)if(ef>=k&&ef<=capacity)configs.emplace_back(k,ef);
        std::mt19937 rng(100+threads);std::shuffle(configs.begin(),configs.end(),rng);
        for(auto [k,ef]:configs){
            std::vector<unsigned> output(nq*k);
            auto batch=[&](size_t query_count){
                #pragma omp parallel for schedule(static) num_threads(threads)
                for(size_t q=0;q<query_count;++q)index.search(queries.data()+q*dimension,k,ef,output.data()+q*k,*scratch[omp_get_thread_num()]);
            };
            if(!verify)batch(std::min(nq,size_t(1000)));
            std::vector<double> seconds;std::vector<size_t> batches;
            for(int sample=0;sample<1;++sample){
                const auto tick=Clock::now();size_t count=0;double duration;
                do{batch(nq);++count;duration=elapsed(tick);}while(!verify&&duration<0.12);
                seconds.push_back(duration);batches.push_back(count);
            }
            size_t correct=0;
            for(size_t q=0;q<nq;++q){
                std::vector<unsigned> ids(output.begin()+q*k,output.begin()+(q+1)*k);
                std::sort(ids.begin(),ids.end());
                if(ids.back()>=unsigned(n)||std::adjacent_find(ids.begin(),ids.end())!=ids.end())throw std::runtime_error("Invalid/duplicate result IDs");
                for(size_t j=0;j<k;++j)correct+=std::binary_search(ids.begin(),ids.end(),gt[q*gt_width+j]);
            }
            const double recall=double(correct)/(nq*k);
            for(size_t sample=0;sample<seconds.size();++sample)
                std::cout<<"PERF {\"kind\":\"query\",\"variant\":\""<<variant<<"\",\"threads\":"<<threads<<",\"k\":"<<k<<",\"ef\":"<<ef
                         <<",\"recall\":"<<recall<<",\"sample\":"<<sample<<",\"nq\":"<<nq<<",\"batches\":"<<batches[sample]
                         <<",\"seconds\":"<<seconds[sample]<<",\"qps\":"<<(nq*batches[sample]/seconds[sample])<<"}\n";
            if(verify){std::ofstream out("ids-k"+std::to_string(k)+"-ef"+std::to_string(ef)+".bin",std::ios::binary);out.write(reinterpret_cast<char*>(output.data()),output.size()*4);}
        }
    }
    return 0;
}catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
