import numpy as np
import math

class HwyChannelLargeScaleFadingGenerator():
    def __init__(self):
        self.stdV2I = 8
        self.stdV2V = 3
        self.vehHeight = 1.5
        self.bsHeight = 25
        self.fc = 2
        self.vehAntGain = 3
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehNoiseFigure = 9
        self.eveAntGain = 3
        self.eveNoiseFigure = 9

        
    # 3GPP37.885 RMa LOS
    def generate_fading_V2I(self, dist_veh2bs, detal_dist=100/3.6*0.01):
        
        d2D = dist_veh2bs
        d3D = np.sqrt((self.vehHeight - self.bsHeight)**2 + dist_veh2bs**2)
        d_bp = 2 * math.pi * self.vehHeight * self.bsHeight * self.fc * (1e9) / (3e8)
        
        def PL1 (d3D):
            return 20 * np.log10(40*math.pi*d3D*self.fc / 3) + min(0.03 * 5 ** 1.72, 10) * np.log10(d3D) - min(0.044 * 5 ** 1.72, 14.77) + 0.002 * d3D * np.log10(5)
        
        def PL2(d_bp, d3D):
            return PL1(d_bp) + 40 * np.log10(d3D / d_bp)
        
        if d2D >= 10 and d2D <= d_bp:
            pathloss = PL1(d3D)
        elif d2D >= d_bp and d2D <= 10 * 1000:
            pathloss = PL2(d3D)
        shadow = np.random.randn() * 4
        combinedPL = -(shadow + pathloss)

        return combinedPL + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure
    
    
    

    def generate_fading_V2V(self, dist_DuePair, detal_dist=100/3.6*0.01):
        if dist_DuePair == 0:
            pathloss = 50
        else:
            if dist_DuePair < 3:  # This condition will barely happen. Just to make sure we do not end up in NaN.
                dist_DuePair = 3
            pathloss = 32.4 + 20 * np.log10(dist_DuePair) + 20 * np.log10(2)
            
        
        shadow = np.random.randn() * self.stdV2V
        combinedPL = -(shadow + pathloss)
        return combinedPL + self.vehAntGain * 2 - self.vehNoiseFigure
    
    def generate_fading_VehicleEve(self, dist_DuePair, detal_dist=100/3.6*0.01):
        if dist_DuePair < 3:  # This condition will barely happen. Just to make sure we do not end up in NaN.
            dist_DuePair = 3
        pathloss = 32.4 + 20 * np.log10(dist_DuePair) + 20 * np.log10(2)
        
        shadow = np.random.randn() * self.stdV2V
        combinedPL = -(shadow + pathloss)
        return combinedPL + self.vehAntGain + self.eveAntGain - self.eveNoiseFigure


class HwyChannelLargeScaleFadingGenerator2():
    def __init__(self):
        self.stdV2I = 8
        self.stdV2V = 3
        self.vehHeight = 1.5
        self.bsHeight = 25
        self.freq = 2
        self.vehAntGain = 3
        self.bsAntGain = 8
        self.bsNoiseFigure = 5
        self.vehNoiseFigure = 9
        self.eveAntGain = 3
        self.eveNoiseFigure = 9

    def generate_fading_V2I(self, dist_veh2bs):
        dist2 = (self.vehHeight - self.bsHeight) ** 2 + dist_veh2bs ** 2
        pathloss = 128.1 + 37.6 * np.log10(np.sqrt(dist2) / 1000)  # 路损公式中距离使用km计算
        combinedPL = -(np.random.randn() * self.stdV2I + pathloss)
        return combinedPL + self.vehAntGain + self.bsAntGain - self.bsNoiseFigure

    def generate_fading_V2V(self, dist_DuePair):
        d_bp = 4 * (self.vehHeight - 1) * (self.vehHeight - 1) * self.freq * (10 ** 9) / (3 * 10 ** 8)
        A = 22.7
        B = 41.0
        C = 20
        if dist_DuePair <= 3:
            pathloss = A * np.log10(3) + B + C * np.log10(self.freq / 5)
        elif dist_DuePair <= d_bp:
            pathloss = A * np.log10(dist_DuePair) + B + C * np.log10(self.freq / 5)
        else:
            pathloss = 40 * np.log10(dist_DuePair) + 9.45 - 17.3 * np.log10(
                (self.vehHeight - 1) * (self.vehHeight - 1)) + 2.7 * np.log10(self.freq / 5)

        combinedPL = -(np.random.randn() * self.stdV2V + pathloss)

        return combinedPL + self.vehAntGain * 2 - self.vehNoiseFigure

    def generate_fading_VehicleEve(self, dist_DuePair):
        d_bp = 4 * (self.vehHeight - 1) * (self.vehHeight - 1) * self.freq * (10 ** 9) / (3 * 10 ** 8)
        A = 22.7
        B = 41.0
        C = 20
        if dist_DuePair <= 3:
            pathloss = A * np.log10(3) + B + C * np.log10(self.freq / 5)
        elif dist_DuePair <= d_bp:
            pathloss = A * np.log10(dist_DuePair) + B + C * np.log10(self.freq / 5)
        else:
            pathloss = 40 * np.log10(dist_DuePair) + 9.45 - 17.3 * np.log10(
                (self.vehHeight - 1) * (self.vehHeight - 1)) + 2.7 * np.log10(self.freq / 5)

        combinedPL = -(np.random.randn() * self.stdV2V + pathloss)
        return combinedPL + self.vehAntGain + self.eveAntGain - self.eveNoiseFigure

if __name__ == '__main__':
    generator = HwyChannelLargeScaleFadingGenerator2()
    print((generator.generate_fading_V2V(20) + 80) / 60)
    print((generator.generate_fading_V2V(20) + 80) / 60)