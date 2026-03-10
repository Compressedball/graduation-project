数据下载网址
https://web.archive.org/web/20190717044203/https://linqs.soe.ucsc.edu/data

cora
adj <class 'scipy.sparse._coo.coo_matrix'> (2708, 2708)
feature <class 'numpy.ndarray'> (2708, 1433) binary
label <class 'numpy.ndarray'> (2708,) [2 5 4 ... 1 0 2] 0~6

citeseer
adj <class 'scipy.sparse._coo.coo_matrix'> (3312, 3312)
feature <class 'numpy.ndarray'> (3312, 3703) binary
label <class 'numpy.ndarray'> (3312,) [1 4 1 ... 4 2 5] 0~5

pubmed
adj <class 'scipy.sparse._coo.coo_matrix'> (19717, 19717)
feature <class 'numpy.ndarray'> (19717, 500)    float32    0.0~1.2633097
label <class 'numpy.ndarray'> (19717,) [0 0 0 ... 0 2 2] 0~2

Q&A?
1.feature是binary或者float是否有影响？
2.有向图到无向图的区别? 转为无向图