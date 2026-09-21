// Adapt the original demo's per-query operations to the common timing harness.
// Every graph/distance kernel is called from the unmodified upstream headers.
struct OriginalIndex {
    int n, dimension, levels, width;
    std::vector<int> lengths, subdims, counts;
    std::vector<std::vector<unsigned char>> merge_data;
    std::vector<std::vector<unsigned char*>> merge_rows;
    std::vector<unsigned char**> merges;
    std::vector<unsigned char> start_merge_data;
    std::vector<unsigned char*> start_merges;
    std::vector<std::vector<char>> flags;
    std::vector<std::vector<unsigned>> transitions;
    std::vector<unsigned> connections;
    std::vector<float> packed;
    hnswlib::L2Space space;
    hnswlib::HierarchicalNSW<float>* graph;
    template<class T> static void read(std::ifstream& in,T* out,size_t count) {
        in.read(reinterpret_cast<char*>(out),count*sizeof(T));
        if (!in) throw std::runtime_error("Truncated upstream index metadata");
    }
    OriginalIndex(int size,int dim,int depth): n(size),dimension(dim),levels(depth),
        width(((dim+(1<<(depth+3))-1)>>(depth+3))<<(depth+3)),
        lengths(depth),subdims(depth),counts(depth),merge_data(depth),merge_rows(depth),merges(depth),
        start_merge_data(4*16*4),start_merges(4),flags(depth),transitions(depth),
        connections(65536*10),space(dim,width) {
        graph=new hnswlib::HierarchicalNSW<float>(&space,"index.bin","index2.bin",false);
        std::ifstream q("quantizer.gt",std::ios::binary);
        read(q,lengths.data(),levels);read(q,subdims.data(),levels);read(q,counts.data(),levels);
        std::ifstream s("searching.gt",std::ios::binary);
        std::vector<float> rotation(size_t(width)*width),codebooks(size_t(width)*256);
        read(s,rotation.data(),rotation.size());
        for(auto& flag:flags){flag.resize(n);read(s,flag.data(),n);}
        for(int level=0;level<levels;++level){
            merge_data[level].resize(lengths[level]*512);
            read(s,merge_data[level].data(),merge_data[level].size());
            merge_rows[level].resize(lengths[level]);
            for(int block=0;block<lengths[level];++block) merge_rows[level][block]=merge_data[level].data()+block*512;
            merges[level]=merge_rows[level].data();
        }
        read(s,start_merge_data.data(),start_merge_data.size());
        for(int block=0;block<4;++block) start_merges[block]=start_merge_data.data()+block*16*4;
        read(s,codebooks.data(),codebooks.size());read(s,connections.data(),connections.size());
        for(int level=0;level<levels;++level){transitions[level].resize(counts[level]);read(s,transitions[level].data(),counts[level]);}
        packed.resize(size_t(width)*(width+256));
        for(int block=0;block<lengths[0];++block){
            float* out=packed.data()+size_t(block)*subdims[0]*(width+256);
            std::copy_n(rotation.data()+size_t(block)*subdims[0]*width,size_t(subdims[0])*width,out);
            std::copy_n(codebooks.data()+block*256*subdims[0],256*subdims[0],out+size_t(subdims[0])*width);
        }
    }
    // The upstream demo keeps the loaded graph alive until process exit. Its
    // destructor is not changed by this harness; retain that lifetime here.
    size_t max_search_queue()const{return *std::min_element(counts.begin(),counts.end());}
    struct Scratch {
        std::vector<float> query,temp,start_data;
        std::vector<float*> start;
        std::vector<std::vector<float>> book_data;
        std::vector<std::vector<float*>> book_rows;
        std::vector<float**> books;
        std::vector<unsigned> entries;
        std::vector<std::vector<elem>> intermediate;
        hnswlib::VisitedListPool visited;
        Scratch(const OriginalIndex& index,size_t capacity):query(index.width,0),temp(index.subdims[0]),
          start_data(64),start(4),book_data(index.levels),book_rows(index.levels),books(index.levels),
          entries(capacity),intermediate(index.levels),visited(1,index.n){
            for(int block=0;block<4;++block)start[block]=start_data.data()+block*16;
            for(int level=0;level<index.levels;++level){
                book_data[level].resize(index.lengths[level]*256);
                book_rows[level].resize(index.lengths[level]);
                for(int block=0;block<index.lengths[level];++block)book_rows[level][block]=book_data[level].data()+block*256;
                books[level]=book_rows[level].data();intermediate[level].resize(capacity);
            }
        }
    };
    void search(const float* query,size_t k,size_t ef,unsigned* out,Scratch& scratch){
        std::copy_n(query,dimension,scratch.query.data());
        std::fill(scratch.start_data.begin(),scratch.start_data.end(),0);
        for(int block=0;block<lengths[0];++block)
            graph->restore_index(scratch.query.data(),packed.data()+size_t(block)*subdims[0]*(width+256),
                scratch.books[0][block],scratch.start.data(),merges.data(),start_merges.data(),
                lengths.data(),subdims.data(),width,scratch.temp.data());
        restore_index2(scratch.query.data(),packed.data(),scratch.books.data(),scratch.start.data(),
            merges.data(),start_merges.data(),lengths.data(),subdims.data(),width,scratch.temp.data(),levels);
        int cell=0;
        for(int block=0;block<4;++block){
            int best=0;float value=scratch.start[block][0];
            for(int center=1;center<16;++center)if(scratch.start[block][center]<value){best=center;value=scratch.start[block][center];}
            cell=cell*16+best;
        }
        unsigned points[10];std::copy_n(connections.data()+cell*10,10,points);
        if(levels==1)graph->SearchWithsingleGraph(scratch.books[0],lengths[0],points,scratch.entries.data(),transitions[0].data(),ef,&scratch.visited);
        else{
            graph->SearchWithquanGraph(scratch.books[levels-1],lengths[levels-1],points,scratch.intermediate[levels-2].data(),
                transitions[levels-1].data(),ef,flags[levels-1].data(),levels-1,&scratch.visited);
            for(int level=levels-2;level>=1;--level)graph->SearchWithquanGraph3(scratch.books[level],lengths[level],
                scratch.intermediate[level].data(),scratch.intermediate[level-1].data(),transitions[level].data(),ef,flags[level].data(),level,&scratch.visited);
            graph->SearchWithquanGraph2(scratch.books[0],lengths[0],scratch.intermediate[0].data(),scratch.entries.data(),transitions[0].data(),ef,&scratch.visited);
        }
        graph->SearchWithOptGraph(query,k,ef,out,scratch.entries.data(),&scratch.visited);
    }
};
