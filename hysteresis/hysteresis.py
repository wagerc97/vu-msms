from pathlib import Path

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


# energy function
def energy(phi, Hpar, Hperp):
    """
    This is the physically correct, dimensional form of the energy.
    Each term has units of energy per volume [J/m³].
    The applied magnetic field has two components: Hpar and Hperp

    :param phi: angle of magnetization of material in a plane
    :param Hpar: parallel to the material’s “easy” direction (usually aligned with the crystal structure)
    :param Hperp: perpendicular to the easy-axis
    :return:
    """
    energy_anisotropy = K * np.sin(phi) ** 2
    energy_zeeman = - mu0 * Ms * (Hpar * np.cos(phi) + Hperp * np.sin(phi))
    total_energy = energy_anisotropy + energy_zeeman
    return total_energy

# energy function
def energy_norm_dimless(phi, Hpar, Hperp):
    """
    This is a normalized, dimensionless version of the energy.

    All energy terms have been divided by K (the anisotropy constant).
    Also, the field values Hpar and Hperp are assumed to be normalized by 2K / (mu0 * Ms) — this helps simplify calculations and comparisons.

    :param phi: angle of magnetization of material in a plane
    :param Hpar: parallel to the material’s “easy” direction (usually aligned with the crystal structure)
    :param Hperp: perpendicular to the easy-axis
    :return:
    """
    return np.sin(phi) ** 2 - Hpar*np.cos(phi) - Hperp*np.sin(phi)


def calc(K, theta_degree):
    """
    Calculate hysteresis.
    :param K: material parameter: anisotropy constant
    :param theta_degree: angle of external field relative to material's easy axis
    :return:
    """

    # Critical field Hc in A/m — tells us when the magnetic field overcomes the material's resistance to magnetization flipping
    # aka "coercive field"
    Hc = 2 * K / (mu0 * Ms)  # [A/m] — "coercive field" (threshold to reorient magnetic domains)
    print(f"Hc: {Hc:.2f} (A/m)")
    
    # Convert to Tesla: B = mu0 * H
    mu0Hc = mu0 * Hc        # [T] — equivalent magnetic flux density
    print(f"Hc: {mu0Hc:.2f} (T)")
    
    # Define a maximum field range — 3x stronger than coercive field to ensure full magnetization switching
    Hmax = 3e6     # [A/m]
    
    # Angle of external field relative to the material's "easy" axis
    theta = np.radians(theta_degree)  # [radians]
    
    #> Generate fields
    #> External field's intensity changes
    # EXPLANATION: 3x definitely above
    Hs = np.concatenate((np.linspace(-Hmax, Hmax, 100), np.linspace(Hmax, -Hmax, 100)))
    
    magns = []  # list to store magnetization projection (dimensionless, normalized)
    
    phi = 0.    # (radians) - initial angle of magnetization

    # Simulate hysteresis
    for H in Hs:
        """
        This loop computes the equilibrium angle of the magnetization (phi) for each external field strength H, 
        and then records how aligned it is with the field — creating a hysteresis loop.
        """
        Hpar  = H * np.cos(theta)        # [A/m] — field along easy axis
        Hperp = H * np.sin(theta)        # [A/m] — field perpendicular to easy axis
    
        # Find angle phi that minimizes energy
        phi = minimize(energy, phi + epsilon, args = (Hpar, Hperp)).x[0]
    
        # Project magnetization along field direction
        # cos(theta - phi): dot product of external field direction and magnetization direction
        magns.append(np.cos(theta - phi))  # dimensionless (since cosines are unitless)
    
    
    # make plot
    #label = f"K={K:.0f}"+r"$\theta$"+f"={theta_degree}°"
    label = f"Hc: {Hc:.2f} (A/m)"
    label = f"K: {K:.0f}"
    plt.plot(Hs, magns, label=label)


if __name__ == '__main__':

    # small perturbation to escape meta-stable point
    epsilon = 1e-8

    # Angle of external field relative to the material's "easy" axis
    theta_degree = 45.   # [degrees]

    # material constants
    mu0 = 4 * np.pi * 1e-7  # [T·m/A] (Tesla meter per Ampere) — magnetic permeability of free space
    Ms = 8e5                # [A/m] — saturation magnetization (how strongly magnetized it can get)
    # K = 1e4               # [J/m^3] (Joule per cubic meter) — anisotropy constant (tells how strongly the material “prefers” certain directions)
    Ks = [1e4, 1e5, 1e6, 1e7]  # vary the anisotropy constant
    Ks = np.logspace(4, 6, 6)
    print(f"Ks: {Ks}")

    for K in Ks:
        calc(K, theta_degree)

    plt.grid()
    plt.legend()
    plt.xlabel("H (A/m)")
    plt.ylabel("M")
    #plt.title(f"Hysteresis loop (" + r"$\theta$" + f"={theta_degree}° | K={K} (J/m³) | " + r"$\epsilon$" + f"={epsilon})")
    plt.title("Hysteresis loops with different K (J/m³)")
    plt.tight_layout()
    filename = f"hystloop-theta={theta_degree}.png"
    #filename = "loops.png"
    plt.savefig(Path("./figs").resolve() / filename)
    plt.show()
    plt.close()
