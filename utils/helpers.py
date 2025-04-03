import numpy as np


'''
Bagian ini adalah helper function yang digunakan untuk membuat kernel Gaussian, saya menggunakan filter gausin supaya lebih mudah saja, dan mengapa saya menggunakan filter ini??
'''
def create_gaussian_kernel(output_channels, input_channels=1):
    '''
    :params output_channels: jumlah channel output
    :params input_channels: jumlah channel input
    '''
    base_kernel = np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ]) / 16.0

    '''
    Menambahkan dimensi untuk kernel di bagian akhir (axis=2) untuk menyesuaikan dengan jumlah channel output,
    akan menjadi mariks dengan ukuran (3,3,1 [Sebab di tambahkan ])
    '''
    kernel = np.expand_dims(base_kernel, axis=2)
    print(f"Kernel Expand : {kernel}")

    kernel = np.repeat(kernel[:, :, np.newaxis, :], output_channels, axis=3)
    print(f"Kernel repeat : {kernel}")


    '''
    Kenapa Bagian Expand, dan Repeat ini penting?
    ================================================================
    Filter ini biasa digunakan saat kita mencari sebuah feature dari gambar (input => hidden),
    jadi kita memerlukan shape filter yang sesuai pula, berikut shape yang digunakna dalam proses ini
    (kernel_height, kernel_width, input_channels, output_channels)
    ---------------------------------------------------------------
    EXP : 
    Shape  kernel : (3,3)
    Feature yang diambil (Extrak) : 16 (per feature memelukan filter untuk membuatnya ), dan,
    setiap filter berkeja di setiap input chanel (Jik RGB maka (3,3,3,16))
    ---------------------------------------------------------------
    => expand_dims: bikin base kernel jadi punya "depth" → bisa dipakai untuk input multi-channel.
    => repeat: menduplikasi kernel untuk setiap output channel → karena kita butuh satu kernel
        untuk tiap output feature map.
    ================================================================
    '''

    if input_channels > 1:
        kernel = np.repeat(kernel, input_channels, axis=2)

    return kernel
