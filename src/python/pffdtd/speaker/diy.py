# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2025 Tobias Hienzsch
import click
import pandas as pd

from pffdtd.common.voltage import dBV_to_volts, dBu_to_dBV


@click.command(name='diy', help='DIY Speakers')
def main():
    price = {
        'dayton_audio_rss315': 315,
        'dayton_audio_rss390': 390,
        'dayton_audio_rss460': 560,
        'tad_et_703a': 5400,
        'tad_td_4001': 5395,
        'tad_td_2002': 3295,
        'tad_tl_1601b': 1925,
        'tad_tl_1801': 2195,
        'radian_950_neopb_8': 533,

        'supravox_285_gmf': 329,
        'supravox_400_gmf': 749,
        'supravox_285_exc': 1239,
        'supravox_400_exc': 1359,
        'volt_vm527': 280,
        'volt_vm752': 669,
        'volt_rv_3143': 569,
        'volt_rv_3863': 779,
        'morel_st_1108': 700/2,
        'morel_tsct_1044': 850/2,
        'scan_speak_32w_4878t00': 700,
        'oberton_nd72ct_hb': 380,

        'bryston_9B_3ch': 13339,
        'bryston_9B_4ch': 16009,
    }

    quattro_t15 = price['tad_td_4001']+price['tad_tl_1601b']*4
    quattro_d15 = price['radian_950_neopb_8']+price['dayton_audio_rss390']*4
    quattro_d12 = price['radian_950_neopb_8']+price['dayton_audio_rss315']*4
    quattro_s12 = price['radian_950_neopb_8']+price['scan_speak_32w_4878t00']*4

    duo_t15 = price['tad_td_4001']+price['tad_tl_1601b']*2
    duo_d15 = price['radian_950_neopb_8']+price['dayton_audio_rss390']*2
    duo_d12 = price['radian_950_neopb_8']+price['dayton_audio_rss315']*2
    duo_s12 = price['radian_950_neopb_8']+price['scan_speak_32w_4878t00']*2

    toby_312dvm = price['dayton_audio_rss315']+price['volt_vm752']+price['morel_tsct_1044']
    toby_312svm = price['scan_speak_32w_4878t00']+price['volt_vm752']+price['morel_tsct_1044']

    toby_315dvm = price['dayton_audio_rss390']+price['volt_vm752']+price['morel_tsct_1044']
    toby_315tvm = price['tad_tl_1601b']+price['volt_vm752']+price['morel_tsct_1044']

    toby_sub12s = price['scan_speak_32w_4878t00']
    toby_sub12d = price['dayton_audio_rss315']
    toby_sub15d = price['dayton_audio_rss390']
    toby_sub15t = price['tad_tl_1601b']
    toby_sub18d = price['dayton_audio_rss460']
    toby_sub18t = price['tad_tl_1801']

    print(pd.DataFrame.from_records([
        {'name': 'Quattro-T15', '1x': quattro_t15, '2x': quattro_t15*2},
        {'name': 'Quattro-S12', '1x': quattro_s12, '2x': quattro_s12*2},
        {'name': 'Quattro-D15', '1x': quattro_d15, '2x': quattro_d15*2},
        {'name': 'Quattro-D12', '1x': quattro_d12, '2x': quattro_d12*2},

        {'name': 'Duo-T15', '1x': duo_t15, '2x': duo_t15*2},
        {'name': 'Duo-S12', '1x': duo_s12, '2x': duo_s12*2},
        {'name': 'Duo-D15', '1x': duo_d15, '2x': duo_d15*2},
        {'name': 'Duo-D12', '1x': duo_d12, '2x': duo_d12*2},

        {'name': '312-SVM', '1x': toby_312svm, '2x': toby_312svm*2},
        {'name': '312-DVM', '1x': toby_312dvm, '2x': toby_312dvm*2},
        {'name': '315-DVM', '1x': toby_315dvm, '2x': toby_315dvm*2},
        {'name': '315-TVM', '1x': toby_315tvm, '2x': toby_315tvm*2},

        {'name': 'S12-D', '1x': toby_sub12d, '2x': toby_sub12d*2},
        {'name': 'S15-D', '1x': toby_sub15d, '2x': toby_sub15d*2},
        {'name': 'S18-D', '1x': toby_sub18d, '2x': toby_sub18d*2},
        {'name': 'S12-S', '1x': toby_sub12s, '2x': toby_sub12s*2},
        {'name': 'S15-T', '1x': toby_sub15t, '2x': toby_sub15t*2},
        {'name': 'S18-T', '1x': toby_sub18t, '2x': toby_sub18t*2},
        {'name': 'S15-T2', '1x': toby_sub15t*2, '2x': toby_sub15t*4},
        {'name': 'S18-T2', '1x': toby_sub18t*2, '2x': toby_sub18t*4},
    ]).to_markdown(index=False))

    # 884 x 498 x 568mm
    height = 0.884
    width = 0.498
    depth = 0.568
    thickness = 25/1000
    density = 600

    box_volume = (width-thickness*2)*(height-thickness*2)*(depth-thickness*2)
    mdf_area = (height*width*2)+(height*depth*2)+(width*depth*2)
    mdf_volume = mdf_area*thickness
    mdf_weight = mdf_volume*density

    driver_weight = 9+12+0.5

    print('------------------')
    print(f'Box Volume    = {box_volume:.3f}m^3')
    print(f'MDF Area      = {mdf_area:.3f}m^2')
    print(f'MDF Volume    = {mdf_volume:.3f}m^3')
    print(f'MDF Weight    = {mdf_weight:.3f}kg')
    print(f'Driver Weight = {driver_weight:.3f}kg')
    print(f'Total Weight  = {mdf_weight+driver_weight:.3f}kg')

    # interface_dac_dBu = 16
    # crossover_adc_dBu = 22
    # crossover_dac_dBu = 16

    # print('------------------')
    # print(f'Interface DAC = {dBV_to_volts(dBu_to_dBV(interface_dac_dBu)):.2f} V')
    # print(f'Crossover ADC = {dBV_to_volts(dBu_to_dBV(crossover_adc_dBu)):.2f} V')
    # print(f'Crossover DAC = {dBV_to_volts(dBu_to_dBV(crossover_dac_dBu)):.2f} V')
    # print(f'Crossover DAC = {dBV_to_volts(dBu_to_dBV(3.5)):.2f} V')
    # print(f'Crossover DAC = {dBV_to_volts(dBu_to_dBV(4)):.2f} V')
