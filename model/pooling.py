import numpy as np
'''
Pooling adalah proses pengambilan fitur dari gambar, dengan cara mengambil nilai maksimum dari bagian tertentu dari gambar, sekaligus digunakan sebagai pengecilan gambar
'''
def max_pooling(input_data, pool_size=(2, 2), stride=None):
    '''
    params: input_data: data matriks gambar yang akan di olah (numpy array)
    params: pool_size: ukuran dari operasi pooling (height, width)
    params: stride: langkah(seberapa banyak piksel) yang akan digunakan untuk melakukan pooling (height, width)
    '''
    ### Stride Kosong atau None, langkah akan disesuikan dengan ukuran pooling
    if stride is None:
        stride = pool_size

    '''
    operasi jika ukuran input adalah 4 dimensi (batch_size, height, width, channel),
    terdapat batch yang digunakna untuk mengelompokan data, misal 1 batch berisi 32 gambar,
    dan saya menggunakan ........ ???
    '''
    ### Merupakan array komprehensif, jadi setiap gambar akan di ambil dari perulangna for dalam batch dan di olah pada bagian ini
    if len(input_data.shape) == 4:
        return np.array([max_pooling(x, pool_size, stride) for x in input_data])

    '''
    h : height dari input_data
    w : width dari input_data
    c : channel dari input_data
    -----------------------------
    ph : height dari pool_size
    pw : width dari pool_size
    -----------------------------
    sh : height dari stride
    sw : width dari stride
    -----------------------------
    out_h : height dari output
    out_w : width dari output
    '''
    h, w, c = input_data.shape
    ph, pw = pool_size
    sh, sw = stride
    out_h = (h - ph) // sh + 1
    out_w = (w - pw) // sw + 1

    '''
    output : hasil dari pooling, yang di inisialisasi dengan 0, dan memiliki ukuran (out_h, out_w, c)
    '''
    output = np.zeros((out_h, out_w, c))

    '''
    dalam proses loopig ini menggunakna panjang dari output, bertujuan supaya operasinya
    sesuai dengan ukuran dari output, dan tidak ada error saat melakukan operasi
    dan ini lah alansan mengapa kita harus mengetahuii ukruan output nya terlbeih dahulu
    ----------------------------------------------------------------------------------
    jadikan ligikanya digunakan atau di buat dengan menggunakan perulangan panjang outputini
    karena ukuran pooling yang akan memperkecil gambar jadi kalau semisal kita menggunakan
    panjang dari image input maka akan terjadi error
    '''
    for i in range(out_h):
        for j in range(out_w):
            for ch in range(c):
                '''
                Menentukan awal iterasi, dan  dibagian ini code akan menyesuaikan
                bagian yang di ambil dengan stride kita
                '''
                h_start = i * sh
                w_start = j * sw

                '''
                Jadi kita akan menentukan bagian awal dan akhirnya  dari proses pooling
                atau mengambil data matriks sesaui dengan ukuran pooling dan dilakukan
                perhitungan(operasi) pooling
                '''
                h_end = h_start + ph
                w_end = w_start + pw
                '''
                proses pooling dilakukan 
                '''
                patch = input_data[h_start:h_end, w_start:w_end, ch]
                output[i, j, ch] = np.max(patch)
    return output
