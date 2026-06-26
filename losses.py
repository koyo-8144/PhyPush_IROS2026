import torch
import torch.nn.functional as F

def log_mse_loss(pred, target):
    log_pred = torch.log1p(torch.abs(pred))
    log_target = torch.log1p(torch.abs(target))
    return F.mse_loss(log_pred, log_target)

class PinnLossCalculator:
    def __init__(self, criterion, gravity=9.81):
        self.criterion = criterion
        self.g = gravity

    def net_force_law(self, mass, acceleration, target_force):
        theory = mass * acceleration
        return self.criterion(target_force, theory)

    def friction_law(self, mass, mu, robot_fz, target_fric):
        normal_force = (mass * self.g) - robot_fz
        theory = mu * torch.clamp(normal_force, min=0.0)
        return self.criterion(target_fric, theory)

    def friction_direct_law(self, mu, normal_force_sensor, target_fric):
        theory = mu * torch.clamp(normal_force_sensor, min=0.0)
        return self.criterion(target_fric, theory)
    
    def inertia_consistency_law(self, mass_est, acceleration_raw, net_f_est):
        implied_mass = net_f_est / (acceleration_raw + 1e-6)
        mass_est_expanded = mass_est.expand_as(implied_mass)
        return self.criterion(mass_est_expanded, implied_mass)

    def inertia_acceleration_law(self, mass_est, net_f_raw, acc_raw):
        eps = 1e-6
        acc_theory = net_f_raw / (mass_est + eps)
        return self.criterion(acc_theory, acc_raw)

    def friction_decoupled_law(self, mass_est, mu_est, robot_fz, fric_f_est):
        effective_fric = fric_f_est + (mu_est * robot_fz)
        weight_ceiling = mu_est * mass_est * self.g
        return self.criterion(effective_fric, weight_ceiling.expand_as(effective_fric))

    def net_force_law_smoothed(self, mass_est, acc_raw, net_f_est):
        T = acc_raw.size(1)
        sum_f = torch.sum(net_f_est, dim=1, keepdim=True)
        sum_a = torch.sum(acc_raw, dim=1, keepdim=True)
        implied_mass_window = sum_f / (sum_a + 1e-6)
        return self.criterion(mass_est.expand(-1, T), implied_mass_window.expand(-1, T))

    def friction_law_robust(self, mass_est, mu_est, robot_fz, fric_f_est):
        T = robot_fz.size(1)
        left_side_seq = fric_f_est + (mu_est * robot_fz)
        left_side_mean = left_side_seq.mean(dim=1, keepdim=True)
        right_side = mu_est * mass_est * self.g
        return self.criterion(left_side_mean.expand(-1, T), right_side.expand(-1, T))

    def kinematic_consistency(self, mass_est, mu_est, push_f_lateral, push_f_vertical, acc_raw):
        eps = 1e-6
        normal_force = (mass_est * self.g) - push_f_vertical
        normal_force = torch.clamp(normal_force, min=0.0)
        fric_force = mu_est * normal_force
        acc_theory = (push_f_lateral - fric_force) / (mass_est + eps)
        return self.criterion(acc_theory, acc_raw)

    def kinematic_position_consistency(self, mass_est, mu_est, push_f_lat, push_f_vert, acc_raw, dt=0.01):
        eps = 1e-6
        normal_force = torch.clamp((mass_est * self.g) - push_f_vert, min=0.0)
        acc_theory = (push_f_lat - (mu_est * normal_force)) / (mass_est + eps)
        vel_theory = torch.cumsum(acc_theory * dt, dim=1)
        pos_theory = torch.cumsum(vel_theory * dt, dim=1)
        vel_raw = torch.cumsum(acc_raw * dt, dim=1)
        pos_raw = torch.cumsum(vel_raw * dt, dim=1)
        return self.criterion(pos_theory, pos_raw)

    def inertia_position_law(self, mass_est, net_f_raw, acc_raw, dt=0.01):
        eps = 1e-6
        acc_theory = net_f_raw / (mass_est + eps)
        vel_theory = torch.cumsum(acc_theory * dt, dim=1)
        pos_theory = torch.cumsum(vel_theory * dt, dim=1)
        vel_raw = torch.cumsum(acc_raw * dt, dim=1)
        pos_raw = torch.cumsum(vel_raw * dt, dim=1)
        return self.criterion(pos_theory, pos_raw)