## Subparts SSD

This is an implementation of the Single Shot Object Detector architecture [(Liu et al.)](https://arxiv.org/abs/1512.02325). In particular, this project provides implementations of the Pooling Pyramid Network variation of SSD [(Jin et al.)](https://arxiv.org/abs/1807.03284), which achieves a meanAP comparable to the original version with the substantial reduction in parameter count.

This implementation proposes an architectural adaptation to allow training taking into account object subparts annotations.

The `networks` directory contains a few architecture variations. In particular, there are implementions alternating the base architecture, as well as containing or not our architectural tweak.

This project allows different SSD variations, simply by extending `networks.ssd.SSD` (without the subparts tweak) or `networks.subparts_ssd.SubParts_SSD` (to use the tweak), and simply following the examples of the other files.

The `QR_Code_Experiments` directory contains a series of experiments aiming at using the PPN version for detection of QR Codes in natural scenes. Here are a few exemple results of the best scoring model.

<img src="https://raw.githubusercontent.com/Leonardo-Blanger/subparts_ppn_keras/parts_based_detector/QR_Code_experiments/samples/qr_code_test_16.png" width="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/subparts_ppn_keras/parts_based_detector/QR_Code_experiments/samples/qr_code_test_23.png" width="430" />

<img src="https://raw.githubusercontent.com/Leonardo-Blanger/subparts_ppn_keras/parts_based_detector/QR_Code_experiments/samples/qr_code_test_42.png" width="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/subparts_ppn_keras/parts_based_detector/QR_Code_experiments/samples/qr_code_test_45.png" width="430" />

<img src="https://raw.githubusercontent.com/Leonardo-Blanger/subparts_ppn_keras/parts_based_detector/QR_Code_experiments/samples/qr_code_test_60.png" width="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/subparts_ppn_keras/parts_based_detector/QR_Code_experiments/samples/qr_code_test_78.png" width="430" />

<img src="https://raw.githubusercontent.com/Leonardo-Blanger/subparts_ppn_keras/parts_based_detector/QR_Code_experiments/samples/qr_code_test_88.png" width="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/subparts_ppn_keras/parts_based_detector/QR_Code_experiments/samples/qr_code_test_92.png" width="430" />

<img src="https://raw.githubusercontent.com/Leonardo-Blanger/subparts_ppn_keras/parts_based_detector/QR_Code_experiments/samples/qr_code_test_97.png" width="430" /> <img src="https://raw.githubusercontent.com/Leonardo-Blanger/subparts_ppn_keras/parts_based_detector/QR_Code_experiments/samples/qr_code_test_29.png" width="430" />
