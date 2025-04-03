import numpy as np
# from matplotlib.sphinxext.plot_directive import exception_template


def conv2D(image, kernel, stride=(1, 1), padding=0):
    # if len(image.shape) == 4:
    #     batch_size = image.shape[0]
    #     return np.array([conv2D(img, kernel, stride, padding) for img in image])
   
    '''
    Menambhkan padding pada citra, supaya kernel conv tidak melebih batas ukuran gambar
    '''

    try:

        if padding > 0:
            image = np.pad(image, ((padding, padding), (padding, padding), (0, 0)), mode='constant')

        ### Mengambil Semua ukuran dari gambar, dan dimensinya
        h, w, c = image.shape

        ### Mengambil ukuran dari kernel
        kh, kw = kernel.shape[:2]

        '''
        Menghitung Ukuran dari output,
        ini dihitung terlbih dahulu sebagai range dari perulangan
        supaya looping sesuai dengan output yang seharusnya
        '''
        out_h = (h - kh) // stride[0] + 1
        out_w = (w - kw) // stride[1] + 1
        out_c = kernel.shape[3]

        output = np.zeros((out_h, out_w, out_c))

        '''
        Perulangan untuk operasi kernel filter dengan gambar
        degnan stride yang diberikan
        '''
        for i in range(out_h):
            for j in range(out_w):

                '''
                Proses pengambilan potongan gambar yang sesuai
                dengan ukruan kernel, yang nantinya akan di lakuan operasi
                untuk mendapatkan featurenya, perkalian ini di lakukan 
                antara kernel dan juga patch(potongan gambar)
                '''
                i_pos = i * stride[0]
                j_pos = j * stride[1]
                patch = image[i_pos:i_pos + kh, j_pos:j_pos + kw, :]


                for f in range(out_c):
                    for c_in in range(c):
                        '''
                        Melakukan operasi perkalian antara gambar patch
                        dan kernel untuk mendapatkan nilai output
                        '''
                        output[i, j, f] += np.sum(patch[:, :, c_in] * kernel[:, :, c_in, f])
    finally:
        print("Succes")

    return output
