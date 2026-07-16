import math
import numpy as np
import time
import pandas as pd


def LLA2NED(lat_rad, long_rad, alt_m, lat_target, long_target, alt_target):
    
    
    # Inputs  latrad: latitude of waypoint (rad)
    #         lonrad: longitude of waypoint (rad)
    #         alt: altitude of waypoint (m) 
    #         lat0_target : Origin latitude (rad)
    #         lon0_target : Origin longitude (rad)
    #         alt0_target : Origin altitude (m)
    # Output  P_ned: NED position (m)
    
    # WGS84 elipsoid constsnts
    a = 6378137
    b = 6356752.3142
    f = 1 - (b/a)
    
    # Converting ECEF co-ords to lat, long and alt
    lat = lat_rad
    long = long_rad
    alt = alt_m
    
    # Targets
    lat0 = lat_target
    long0 = long_target
    alt0 = alt_target
    
    #Estimating Constants
    RNi = a/math.sqrt(1 - (2*f - f**2)*math.sin(lat0)**2) # RN
    RMi = RNi*((1-(2*f - f**2))/(1-(2*f - f**2)*math.sin(lat0)**2)) # RM
    
    #Estimating NED from RNi and RMi
    dN = (lat - lat0)/ a*np.arctan(1/RMi)# N
    dE = (long - long0)/(a*np.arctan(1/(RNi*math.cos(lat0)))) #E
    dD = alt0 - alt #D
    
    N = dN
    E = dE
    D = dD
    
    P_NED = (N,E,D)
    return P_NED


    #NE#North East Down to Latitude, Longitude, Altitude Conversion (completed)
def NED2LLA(N, E, D, lat_target, lon_target, alt_target):
    
    # Inputs  N: North position {x} (m)
    #         E: East position {y} (m)
    #         D: Down position {z} (m) 
    #         lat0_target : Origin latitude (rad)
    #         lon0_target : Origin longitude (rad)
    #         alt0_target : Origin altitude (m)
    #
    # Output  P_LLA: LLA position (m)    
    
    # Required constants
    a = 6378137;
    b = 6356752.3142;
    e_sqr = (a**2-b**2)/a**2;
    e2_sqr = (a**2-b**2)/b**2; 
    #
    #D to ECEF rotation matrix using the origin 
    R_NEF = np.zeros([3,3])
    R_NEF[0,0] = -np.sin(lat_target)*np.cos(lon_target)
    R_NEF[0,1] = -np.sin(lon_target)
    R_NEF[0,2] = -np.cos(lat_target)*np.cos(lon_target)
    R_NEF[1,0] = -np.sin(lat_target)*np.sin(lon_target)
    R_NEF[1,1] = np.cos(lon_target)
    R_NEF[1,2] = -np.cos(lat_target)*np.sin(lon_target)
    R_NEF[2,0] = np.cos(lat_target)
    R_NEF[2,1] = 0
    R_NEF[2,2] = -np.sin(lat_target)

    #ECEF origin
    R_N_origin = a/np.sqrt(1-(e_sqr*(np.sin(lat_target))**2));
    O_ECEF = np.zeros([3,1])
    O_ECEF[0,0] = (R_N_origin + alt_target)*np.cos(lat_target)*np.cos(lon_target)
    O_ECEF[1,0] = (R_N_origin + alt_target)*np.cos(lat_target)*np.sin(lon_target)
    O_ECEF[2,0] = ((1-e_sqr)*R_N_origin + alt_target)*np.sin(lat_target)

    #find ECEF X Y Z and offset by the origin
    P_ECEF = R_NEF @ np.array([[N],[E],[D]]) +  O_ECEF;
    
    #convert P_ECEF to LLA (use closed form solution)
    X = P_ECEF[0][0];
    Y = P_ECEF[1][0];
    Z = P_ECEF[2][0];
    LLA_longitude = np.arctan2(Y,X);
    p = np.sqrt(X**2+Y**2);
    theta = np.arctan((Z*a)/(p*b));
    LLA_latitude = np.arctan((Z+e2_sqr*b*(np.sin(theta))**3)/(p-e_sqr*a*(np.cos(theta))**3));
    R_N_LLAlat = a/np.sqrt(1-(e_sqr*(np.sin(LLA_latitude))**2));
    LLA_altitude = p/np.cos(LLA_latitude) - R_N_LLAlat;
    
    #return LLA latitude, longitude and altitude
    P_LLA = (LLA_latitude, LLA_longitude, LLA_altitude);    
    
    return P_LLA