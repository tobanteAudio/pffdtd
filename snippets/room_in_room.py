# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Tobias Hienzsch
import math


def main():
    L = 6
    W = 3.65
    H = 3.12

    L = 8.5
    W = 6.5
    H = 3.5

    floor_area = L*W

    length_wall_area = L*H*2
    width_wall_area = W*H*2
    wall_area = (L*H+W*H)*2

    drywall_density = 800
    drywall_thickness = 0.0125*2
    drywall_surface_mass = drywall_density*drywall_thickness
    drywall_weight = wall_area*drywall_surface_mass

    osb_density = 600
    osb_thickness = 0.025
    osb_surface_mass = osb_density*osb_thickness
    osb_weight = wall_area*osb_surface_mass

    sonorock_width = 0.625
    sonorock_thickness = 0.1
    sonorock_density = 8.75/sonorock_width/sonorock_thickness
    sonorock_surface_mass = sonorock_density*sonorock_thickness
    sonorock_weight = wall_area*sonorock_surface_mass

    stud_width = 0.1
    stud_thickness = 0.06
    stud_density = 470
    stud_count_length = int(math.ceil(L/(sonorock_width+stud_thickness))*2)
    stud_count_width = int(math.ceil(W/(sonorock_width+stud_thickness))*2)
    stud_count = stud_count_width+stud_count_length
    stud_volume = H*stud_width*stud_thickness*stud_count
    stud_weight = stud_volume*stud_density

    floor_stud_width = 0.1
    floor_stud_thickness = 0.1
    floor_stud_count = int(math.ceil(L/(sonorock_width+floor_stud_thickness)))
    floor_stud_volume = W*floor_stud_width*floor_stud_thickness*floor_stud_count
    floor_stud_weight = floor_stud_volume*stud_density
    floor_sonorock_weight = floor_area*sonorock_surface_mass
    floor_osb_weight = floor_area*osb_surface_mass*2

    floor_weight = floor_osb_weight+floor_sonorock_weight+floor_stud_weight
    wall_weight = stud_weight+drywall_weight+osb_weight+sonorock_weight
    extra_weight = 1500
    total_weight = floor_weight+wall_weight+extra_weight

    print(f"Floor Area:            {floor_area:.2f} m^2")
    print(f"Length Wall Area:      {length_wall_area:.2f} m^2")
    print(f"Width Wall Area:       {width_wall_area:.2f} m^2")
    print(f"Total Wall Area:       {wall_area:.2f} m^2")
    print('-'*32)

    print(f"Drywall Surface Mass:  {drywall_surface_mass:.2f} kg/m^2")
    print(f"OSB Surface Mass:      {osb_surface_mass:.2f} kg/m^2")
    print(f"Sonorock Surface Mass: {sonorock_surface_mass:.2f} kg/m^2")
    print('-'*32)

    print(f"Wall Studs Length:     {stud_count_length}")
    print(f"Wall Studs Width:      {stud_count_width}")
    print(f"Wall Studs Total:      {stud_count_length+stud_count_width}")
    print(f"Wall Studs Volume:     {stud_volume:.3f} m^3")
    print('-'*32)

    print(f"Wall Studs Weight:     {stud_weight:.2f} kg")
    print(f"Wall Drywall Weight:   {drywall_weight:.2f} kg")
    print(f"Wall OSB Weight:       {osb_weight:.2f} kg")
    print(f"Wall Sonorock Weight:  {sonorock_weight:.2f} kg")
    print('-'*32)

    print(f"Floor Studs:           {floor_stud_count}")
    print(f"Floor Studs Volume:    {floor_stud_volume:.3f} m^3")
    print(f"Floor Studs Weight:    {floor_stud_weight:.3f} kg")
    print(f"Floor Sonorock Weight: {floor_sonorock_weight:.3f} kg")
    print(f"Floor OSB Weight:      {floor_osb_weight:.3f} kg")
    print('-'*32)

    print(f"Floor Weight:          {floor_weight:.2f} kg")
    print(f"Wall Weight:           {wall_weight:.2f} kg")
    print(f"Extra Weight:          {extra_weight:.2f} kg")
    print(f"Total Weight:          {total_weight:.2f} kg")
    print(f"Weight/Area:           {total_weight/floor_area:.2f} kg/m^2")


if __name__ == '__main__':
    main()
