import matplotlib.pyplot as plt
import pydicom
from pydicom.data import get_testdata_files
import numpy as np
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

base = "/OLD-DATA-STOR/X-RAY-Project/db/completeDatabase/completeData/"
pass_dicom = ['IM-0140-0002.dcm'] #,'10101133.dcm','09460972.dcm','09351741.dcm','10233876.dcm','11002225.dcm','12195703.dcm','13341752.dcm','141.95188.dcm','14100333.dcm','15051834.dcm','17002381.dcm','12195703.dcm','13341752.dcm','14195188.dcm','14100333.dcm','15051834.dcm','15031100.dcm','15490746.dcm','17002381.dcm','IM-0055-0002.dcm','21341556.dcm','IM-0161-0002.dcm','IM-0139-0001.dcm','IM-0120-0002.dcm','IM-0105-0002.dcm','IM-0079-0001-002.dcm','IM-0313-0001.dcm','IM-0310-0001.dcm','IM-0301-0002.dcm','IM-0280-0002.dcm','IM-0279-0001.dcm','IM-0341-0001.dcm','IM-0539-0001.dcm','IM-0630-0002.dcm','IM-0591-0001.dcm','IM-0738-0002.dcm','IM-0705-0001.dcm','IM-0895-0004.dcm','IM-1049-0002.dcm','IM-1132-0001.dcm','IM-1128-0001.dcm','IM-1062-0001-0002.dcm','IM-1057-0002.dcm','IM-1056-0002.dcm','IM-1323-0002.dcm','IM-1273-0003.dcm','IM-1452-0002.dcm','IM-1448-0001-0005.dcm','IM-1360-0002.dcm','IM-1332-0002.dcm','IM-1597-0001.dcm','IM-1594-0002.dcm','IM-1588-0002.dcm','IM-1555-0002.dcm','IM-1540-0002.dcm','IM-1523-0001.dcm','IM-1509-0001.dcm','IM-1736-0002.dcm','IM-1725-0002.dcm','IM-1670-0002.dcm','IM-1649-0003.dcm','IM-1850-0004.dcm','IM-1843-0001.dcm','IM-1808-0001.dcm','IM-1806-0006.dcm','IM-1805-0005.dcm','IM-1804-0002.dcm','IM-1792-0001.dcm','IM-1792-0001.dcm','IM-2036-0004.dcm','IM-2629-0001.dcm','IM-2611-0005.dcm','IM-3329-0002.dcm','IM-6098-0001.dcm','IM-5964-0005.dcm','IM-6216-0001.dcm','IM-6169-0002.dcm','IM-6791-0001.dcm','IM-7533-0003.dcm','IM-7511-0011.dcm','IM-7508-0008.dcm','IM-7507-0007.dcm','IM-7506-0006.dcm','IM-7505-0005.dcm','IM-7347-0001.dcm','IM-7838-0002.dcm','IM-7803-0002.dcm','IM-8738-0002.dcm','IM-8527-0002.dcm','IMG-0328-0001-0004.dcm','IMG-0324-0001.dcm','IMG-0307-0001-0006.dcm','IMG-0176-0001.dcm','IMG-0502-0002.dcm','IMT-0127-0001.dcm','IMT-0113-0002.dcm','IMT-0044-0001.dcm','IMT-0267-0002.dcm','IMT-0159-0001.dcm','IMT-0155-0003.dcm','IMT-0143-0003.dcm','IMT-0139-0001.dcm','IMT-0512-0003.dcm','IMT-0404-0001.dcm','IMT-0404-0001.dcm','IMT-0558-0002.dcm','IMT-0547-0001.dcm','IMT-0916-0001.dcm','IMT-0899-0001.dcm','IMT-0793-0004.dcm','IMT-0776-0002.dcm','IMT-0731-0001.dcm','IMT-0702-0001-0003.dcm','IMT-0961-0002.dcm','SO-0123-0002.dcm','SO-0117-0001-0002.dcm','SO-0229-0001-0002.dcm','SO-0161-0003.dcm','SO-0146-0002.dcm','SO0322-0002.dcm','SO-0302-0002.dcm','SO-0626-0002.dcm','SO-0607-0002.dcm'





#img=get_testdata_files("/OLD-DATA-STOR/X-RAY-Project/db/test set/00063657.dcm")[0]

for a in pass_dicom:
    filenames = pydicom.data.data_manager.get_files(base, a)
    if len(filenames) > 0:
        filename = filenames[0]
        ds = pydicom.dcmread(filename)

        plt.imshow(ds.pixel_array, cmap=plt.cm.bone)
        plt.axis('off')
        # plt.title(f'{a}')
        plt.show()
    else:
        print(f"No DICOM files found for {a}")


    
## temp to show the heatmap, not useful anymore


# img_size = (224,224)

# threeDprocess = transforms.Compose([
#     transforms.Resize(img_size)
# ])


# base = "/OLD-DATA-STOR/X-RAY-Project/db/completeDatabase/completeData/"
# pass_dicom = ['20180064.dcm'] #,'10101133.dcm','09460972.dcm','09351741.dcm','10233876.dcm','11002225.dcm','12195703.dcm','13341752.dcm','141.95188.dcm','14100333.dcm','15051834.dcm','17002381.dcm','12195703.dcm','13341752.dcm','14195188.dcm','14100333.dcm','15051834.dcm','15031100.dcm','15490746.dcm','17002381.dcm','IM-0055-0002.dcm','21341556.dcm','IM-0161-0002.dcm','IM-0139-0001.dcm','IM-0120-0002.dcm','IM-0105-0002.dcm','IM-0079-0001-002.dcm','IM-0313-0001.dcm','IM-0310-0001.dcm','IM-0301-0002.dcm','IM-0280-0002.dcm','IM-0279-0001.dcm','IM-0341-0001.dcm','IM-0539-0001.dcm','IM-0630-0002.dcm','IM-0591-0001.dcm','IM-0738-0002.dcm','IM-0705-0001.dcm','IM-0895-0004.dcm','IM-1049-0002.dcm','IM-1132-0001.dcm','IM-1128-0001.dcm','IM-1062-0001-0002.dcm','IM-1057-0002.dcm','IM-1056-0002.dcm','IM-1323-0002.dcm','IM-1273-0003.dcm','IM-1452-0002.dcm','IM-1448-0001-0005.dcm','IM-1360-0002.dcm','IM-1332-0002.dcm','IM-1597-0001.dcm','IM-1594-0002.dcm','IM-1588-0002.dcm','IM-1555-0002.dcm','IM-1540-0002.dcm','IM-1523-0001.dcm','IM-1509-0001.dcm','IM-1736-0002.dcm','IM-1725-0002.dcm','IM-1670-0002.dcm','IM-1649-0003.dcm','IM-1850-0004.dcm','IM-1843-0001.dcm','IM-1808-0001.dcm','IM-1806-0006.dcm','IM-1805-0005.dcm','IM-1804-0002.dcm','IM-1792-0001.dcm','IM-1792-0001.dcm','IM-2036-0004.dcm','IM-2629-0001.dcm','IM-2611-0005.dcm','IM-3329-0002.dcm','IM-6098-0001.dcm','IM-5964-0005.dcm','IM-6216-0001.dcm','IM-6169-0002.dcm','IM-6791-0001.dcm','IM-7533-0003.dcm','IM-7511-0011.dcm','IM-7508-0008.dcm','IM-7507-0007.dcm','IM-7506-0006.dcm','IM-7505-0005.dcm','IM-7347-0001.dcm','IM-7838-0002.dcm','IM-7803-0002.dcm','IM-8738-0002.dcm','IM-8527-0002.dcm','IMG-0328-0001-0004.dcm','IMG-0324-0001.dcm','IMG-0307-0001-0006.dcm','IMG-0176-0001.dcm','IMG-0502-0002.dcm','IMT-0127-0001.dcm','IMT-0113-0002.dcm','IMT-0044-0001.dcm','IMT-0267-0002.dcm','IMT-0159-0001.dcm','IMT-0155-0003.dcm','IMT-0143-0003.dcm','IMT-0139-0001.dcm','IMT-0512-0003.dcm','IMT-0404-0001.dcm','IMT-0404-0001.dcm','IMT-0558-0002.dcm','IMT-0547-0001.dcm','IMT-0916-0001.dcm','IMT-0899-0001.dcm','IMT-0793-0004.dcm','IMT-0776-0002.dcm','IMT-0731-0001.dcm','IMT-0702-0001-0003.dcm','IMT-0961-0002.dcm','SO-0123-0002.dcm','SO-0117-0001-0002.dcm','SO-0229-0001-0002.dcm','SO-0161-0003.dcm','SO-0146-0002.dcm','SO0322-0002.dcm','SO-0302-0002.dcm','SO-0626-0002.dcm','SO-0607-0002.dcm'





# #img=get_testdata_files("/OLD-DATA-STOR/X-RAY-Project/db/test set/00063657.dcm")[0]

# for a in pass_dicom:
#     filenames = pydicom.data.data_manager.get_files(base, a)
#     if len(filenames) > 0:
#         filename = filenames[0]
#         ds = pydicom.dcmread(filename)
#         dcs=np.array(ds.pixel_array, dtype=np.float32) / np.max(ds.pixel_array)
#         dct=np.float32(threeDprocess(Image.fromarray(dcs)))
#         dct=cv2.merge([dct, dct, dct])
#         grayscale_cam = np.array(Image.open('/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Code/output/ablationcam_heatmap.jpg'))
#         grayscale_came=np.divide(grayscale_cam, np.max(grayscale_cam))
#         print(grayscale_cam)

#         mask = grayscale_came > 0.6

#         # Apply the mask to the array to filter out smaller values
#         grayscale_camm = np.where(mask, grayscale_came, 0)


#         # Define the Gaussian blur parameters
#         kernel_size = (15, 15)  # Size of the Gaussian kernel
#         sigma_x = 5             # Standard deviation along X-axis (automatically calculated based on kernel size)
#         # Apply Gaussian blur
#         grayscale_camm = cv2.GaussianBlur(grayscale_camm, kernel_size, sigma_x)
#         grayscale_camt=np.array(grayscale_camm*255).astype(np.uint8)
#         grayscale_cammap = cv2.applyColorMap(grayscale_camt, cv2.COLORMAP_TURBO)
#         plt.imshow(Image.fromarray(grayscale_camt))
#         #cam_image = show_cam_on_image(dct, grayscale_camm, use_rgb=True, colormap = cv2.COLORMAP_TURBO, image_weight = 0.85)

#         plt.imshow(dct**0.5, alpha=0.85)
#         plt.axis('off')
#         plt.show()
#         cam_output_path = '/OLD-DATA-STOR/HESSO_Internship_2023/Piotr/Code/Visualization/test.jpg'
#         #cv2.imwrite(cam_output_path, cam_image)
#     else:
#         print(f"No DICOM files found for {a}")