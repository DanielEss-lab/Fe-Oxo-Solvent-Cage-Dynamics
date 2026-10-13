import os
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from .model import Generator, Discriminator


class BGANEPS(object):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def generator_loss(self, z, x, epoch, epochs):
        x_ = self.g_net(z)
        z_ = self.e_net(x)

        x__ = self.g_net(z_)
        z__ = self.e_net(x_)

        if epoch + 1 > epochs - 30:
            generated = self.reconstruct(x__)
            if len(generated) > 0:
                self.dofs = np.vstack((self.dofs, generated))
                self.real_mask = np.hstack(
                    (self.real_mask, np.zeros(len(generated), dtype=bool))
                )

        dz_ = self.dz_net(z_)
        dx_ = self.dx_net(x_)

        l2_loss_z = self.mse_loss(z, z__)
        l2_loss_x = self.mse_loss(x, x__)

        g_loss_adv = self.mse_loss(torch.ones_like(dz_), dz_)
        e_loss_adv = self.mse_loss(torch.ones_like(dx_), dx_)

        g_loss = l2_loss_x
        e_loss = l2_loss_z
        g_e_loss = g_loss_adv + e_loss_adv + 10.0 * l2_loss_z + 10.0 * l2_loss_x
        return g_loss, e_loss, g_e_loss

    def discriminator_loss(self, z, x):
        x_ = self.g_net(z)
        z_ = self.e_net(x)

        dz = self.dz_net(z)
        dx = self.dx_net(x)

        dz_ = self.dz_net(z_)
        dx_ = self.dx_net(x_)

        dz_loss = (
            self.mse_loss(torch.ones_like(dz), dz)
            + self.mse_loss(torch.zeros_like(dz_), dz_)
        ) / 2.0
        dx_loss = (
            self.mse_loss(torch.ones_like(dx), dx)
            + self.mse_loss(torch.zeros_like(dx_), dx_)
        ) / 2.0

        d_loss = dz_loss + dx_loss
        return dz_loss, dx_loss, d_loss

    def reconstruct(self, X):
        X = X.detach().cpu().numpy()
        X = self.scaler.inverse_transform(X)
        lower, upper = sorted((self.bondmin, self.bondmax))
        X = X[(X[:, 0] < upper) & (X[:, 0] > lower)]
        return X

    def write_validation_report(self, args, eps_output_dir):
        real = self.dofs[self.real_mask]
        generated = self.dofs[~self.real_mask]
        lines = [
            "BGAN-EPS validation summary",
            f"input_dim: {self.dim}",
            f"latent_dim: {self.latent_dim}",
            f"hidden_nodes: {self.hidden_nodes}",
            f"batch_size: {self.effective_batch_size}",
            f"epochs: {args.epochs}",
            f"learning_rate: {args.lr}",
            f"real_samples_after_rc_filter: {len(real)}",
            f"generated_samples_after_rc_filter: {len(generated)}",
            f"augmented_samples_total: {len(self.dofs)}",
            f"bins: {args.bins}",
            f"window_points: {args.window_points}",
            f"rc_direction: {args.rc_direction}",
            "",
        ]
        if len(generated) > 0 and len(real) > 0:
            mean_diff = np.abs(real.mean(axis=0) - generated.mean(axis=0))
            std_diff = np.abs(real.std(axis=0) - generated.std(axis=0))
            lines.extend(
                [
                    "Real/generated distribution checks across internal-coordinate columns",
                    f"mean_abs_mean_diff: {mean_diff.mean():.6f}",
                    f"max_abs_mean_diff: {mean_diff.max():.6f}",
                    f"mean_abs_std_diff: {std_diff.mean():.6f}",
                    f"max_abs_std_diff: {std_diff.max():.6f}",
                ]
            )
        else:
            lines.append("Real/generated distribution checks skipped because one set is empty.")

        with open(os.path.join(eps_output_dir, "validation_summary.txt"), "w") as fw:
            fw.write("\n".join(lines) + "\n")

        if args.save_augmented:
            np.save(os.path.join(eps_output_dir, "augmented_dof.npy"), self.dofs)
            np.save(os.path.join(eps_output_dir, "augmented_real_mask.npy"), self.real_mask)

    def run(self, args, X, dim, scaler, loop):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dim = dim
        self.latent_dim = args.latent_dim if args.latent_dim > 0 else dim
        self.hidden_nodes = args.hidden_nodes if args.hidden_nodes > 0 else dim
        self.g_net = Generator(input_dim=self.latent_dim, output_dim=dim, nb_units=self.hidden_nodes).to(
            self.device
        )
        self.e_net = Generator(input_dim=dim, output_dim=self.latent_dim, nb_units=self.hidden_nodes).to(
            self.device
        )
        self.dz_net = Discriminator(input_dim=self.latent_dim, nb_units=max(1, self.latent_dim // 4)).to(self.device)
        self.dx_net = Discriminator(input_dim=dim, nb_units=max(1, dim // 4)).to(self.device)
        self.mse_loss = torch.nn.MSELoss()
        self.scaler = scaler
        self.bondmax = args.bondmax
        self.bondmin = args.bondmin

        TRAIN = torch.from_numpy(X).float()
        self.dofs = self.reconstruct(TRAIN)
        self.real_mask = np.ones(len(self.dofs), dtype=bool)
        if len(TRAIN) < 2:
            raise ValueError("At least two snapshots are required for BGAN training.")
        self.effective_batch_size = min(args.batch_size, len(TRAIN))
        if self.effective_batch_size < 2:
            raise ValueError("Effective batch size must be at least two.")
        train_set = DataLoader(
            TRAIN, batch_size=self.effective_batch_size, shuffle=True, drop_last=True
        )
        self.g_e_optimizer = torch.optim.Adam(
            list(self.g_net.parameters()) + list(self.e_net.parameters()),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
        )
        self.d_optimizer = torch.optim.Adam(
            list(self.dz_net.parameters()) + list(self.dx_net.parameters()),
            lr=args.lr,
            betas=(args.beta1, args.beta2),
        )
        for epoch in range(args.epochs):
            for i, x in enumerate(train_set):
                z = torch.randn(x.shape[0], self.latent_dim) / 2.0
                z, x = z.to(self.device), x.to(self.device)

                # Generator / Encoder Training
                self.g_e_optimizer.zero_grad()
                g_loss, e_loss, g_e_loss = self.generator_loss(z, x, epoch, args.epochs)
                g_e_loss.backward()
                self.g_e_optimizer.step()

                self.d_optimizer.zero_grad()
                dz_loss, dx_loss, d_loss = self.discriminator_loss(z, x)
                d_loss.backward()
                self.d_optimizer.step()
            print(
                "Epoch [%d] g_loss [%.4f] e_loss [%.4f] g_e_loss [%.4f] dz_loss [%.4f] dx_loss [%.4f] d_loss [%.4f]"
                % (epoch + 1, g_loss, e_loss, g_e_loss, dz_loss, dx_loss, d_loss)
            )

        eps_output_dir = os.path.join(self.output_dir, f"bgan{loop+1}")
        if not os.path.exists(eps_output_dir):
            os.makedirs(eps_output_dir)
        self.write_validation_report(args, eps_output_dir)

        real_mask = self.real_mask
        self.dofs = self.dofs.T
        self.real_mask = real_mask
        self.EPS(args, loop)

    def get_jacobian(self, X_bin_edge, xjtype, Y_bin_edge, yjtype):
        if Y_bin_edge is None and yjtype is None:
            if xjtype == "bond":
                return X_bin_edge**2
            elif xjtype == "torsion":
                return 1
            else:
                return math.sin(X_bin_edge)
        else:
            return_dict = {
                ("bond", "bond"): X_bin_edge**2 * Y_bin_edge**2,
                ("bond", "angle"): X_bin_edge**2 * math.sin(Y_bin_edge),
                ("bond", "torsion"): X_bin_edge**2,
                ("angle", "angle"): math.sin(X_bin_edge) * math.sin(Y_bin_edge),
                ("angle", "torsion"): math.sin(X_bin_edge),
                ("torsion", "bond"): Y_bin_edge**2,
                ("torsion", "angle"): math.sin(Y_bin_edge),
            }
            return return_dict.get((xjtype, yjtype), 1)

    def entropy_1D(self, args):
        num, jtype = args
        counts, bin_edges = np.histogram(self.dofs[num][self.indices], self.bins)
        sample_size = np.sum(counts)
        if sample_size == 0:
            return np.nan
        prob_den = counts / sample_size
        dx = bin_edges[1] - bin_edges[0]
        if dx == 0:
            return np.nan
        bin_edges += dx / 2

        entropy_sum = -np.sum(
            [
                prob_den[i]
                * math.log(
                    prob_den[i]
                    / (self.get_jacobian(bin_edges[i], jtype, None, None) * dx)
                )
                for i in range(self.bins)
                if prob_den[i] != 0
            ]
        )

        return entropy_sum + (np.count_nonzero(prob_den) - 1) / (2 * sample_size)

    def entropy_2D(self, args):
        xnum, ynum, xjtype, yjtype = args
        H, X_bin_edges, Y_bin_edges = np.histogram2d(
            self.dofs[xnum][self.indices], self.dofs[ynum][self.indices], self.bins
        )
        dx, dy = X_bin_edges[1] - X_bin_edges[0], Y_bin_edges[1] - Y_bin_edges[0]
        if dx == 0 or dy == 0:
            return np.nan
        X_bin_edges += dx / 2
        Y_bin_edges += dy / 2
        sample_size = np.sum(H)
        if sample_size == 0:
            return np.nan
        H = H / sample_size

        entropy_sum = -np.sum(
            [
                H[row][col]
                * math.log(
                    H[row][col]
                    / (
                        self.get_jacobian(
                            X_bin_edges[row], xjtype, Y_bin_edges[col], yjtype
                        )
                        * dx
                        * dy
                    )
                )
                for row in range(self.bins)
                for col in range(self.bins)
                if H[row][col] != 0
            ]
        )

        return entropy_sum + (np.count_nonzero(H) - 1) / (2 * sample_size)

    def get_sorted_indices(self, entropy_list, num_indices):
        return sorted(range(len(entropy_list)), key=lambda i: entropy_list[i])[
            -num_indices:
        ]

    def EPS(self, args, loop):
        if args.eps_type == "max":
            raise NotImplementedError(
                "--eps_type max is not implemented in this Fe-oxo branch; use --eps_type average."
            )
        self.bins = args.bins
        s_profile, sample_size, real_sample_size = [], [], []
        conversion_factor = -1.987204259e-3 * args.temperature  # kcal/mol
        ensemble = args.ensemble + 1
        if args.rc_direction == "decreasing":
            x = np.linspace(args.bondmax, args.bondmin, ensemble)
            dx = x[0] - x[1]
            runavg = np.linspace(x[0], x[-2], args.window_points)
        else:
            x = np.linspace(args.bondmin, args.bondmax, ensemble)
            dx = x[1] - x[0]
            runavg = np.linspace(x[0], x[-2], args.window_points)
        bat = {2: "bond", 3: "angle", 4: "torsion"}
        eps_output_dir = os.path.join(self.output_dir, f"bgan{loop+1}")
        if not os.path.exists(eps_output_dir):
            os.makedirs(eps_output_dir)

        with open(os.path.join(self.output_dir, "topology.txt"), "r") as topology_file:
            topology_lines = topology_file.readlines()
        dof_list = [line.split() for line in topology_lines if len(line.split()) != 1]
        atom_num = int(topology_lines[0])
        bond_num = len([items for items in dof_list if len(items) == 2])
        angle_num = len([items for items in dof_list if len(items) == 3])
        torsion_num = len([items for items in dof_list if len(items) == 4])

        react = self.dofs[0]
        self.dofs = self.dofs[1:]

        for e in range(args.window_points):
            if args.rc_direction == "decreasing":
                low, high = runavg[e] - dx, runavg[e]
            else:
                low, high = runavg[e], runavg[e] + dx
            eps_file_path = f"{eps_output_dir}/eps_{high:.3f}_{low:.3f}.log"
            with open(eps_file_path, "w") as f_eps:
                self.indices = np.argwhere((react > low) & (react < high)).flatten()
                sample_size.append(len(self.indices))
                real_sample_size.append(np.count_nonzero(self.real_mask[self.indices]))
                if len(self.indices) == 0:
                    s_profile.append(np.nan)
                    f_eps.write("Empty structural window; entropy not computed.\n")
                    continue
                if len(self.indices) < args.min_window_samples:
                    f_eps.write(
                        f"WARNING: augmented window has {len(self.indices)} samples, "
                        f"below --min_window_samples={args.min_window_samples}.\n"
                    )
                # with Pool() as p:
                #    entropy1d = p.map(self.entropy_1D, [(i, bat[len(dof_list[i])]) for i in range(len(dof_list))])
                #    entropy2d = p.map(self.entropy_2D, [(i, j, bat[len(dof_list[i])], bat[len(dof_list[j])]) for i in range(len(dof_list) - 1) for j in range(i + 1, len(dof_list))])
                entropy1d = [
                    self.entropy_1D((i, bat[len(dof_list[i])]))
                    for i in range(len(dof_list))
                ]
                entropy2d = [
                    self.entropy_2D(
                        (i, j, bat[len(dof_list[i])], bat[len(dof_list[j])])
                    )
                    for i in range(len(dof_list) - 1)
                    for j in range(i + 1, len(dof_list))
                ]
                for i in range(len(dof_list)):
                    f_eps.write(
                        f'{bat[len(dof_list[i])]} {" ".join(str(x) for x in dof_list[i])}\t{entropy1d[i]}\n'
                    )

                MI_list, MIST_list = [], []
                index = 0
                MI = 0.0
                for i in range(len(dof_list) - 1):
                    dummy = []
                    for j in range(i + 1, len(dof_list)):
                        dummy.append(entropy1d[i] + entropy1d[j] - entropy2d[index])
                        f_eps.write(
                            "%s %s\t%s %s\t\t%f\t%f\n"
                            % (
                                bat[len(dof_list[i])],
                                " ".join(str(x) for x in dof_list[i]),
                                bat[len(dof_list[j])],
                                " ".join(str(x) for x in dof_list[j]),
                                entropy2d[index],
                                dummy[-1],
                            )
                        )
                        index += 1
                    MI_list.append(dummy)
                    MIST_list.append(np.nanmax(dummy))
                    MI += sum(dummy)

                if args.eps_type == "max":
                    f_eps.write("Total MIE: %f\n" % MI)
                    f_eps.write("Total MIST: %f\n" % sum(MIST_list))
                    f_eps.write("MIE Entropy: %f\n" % (sum(entropy1d) - MI))
                    f_eps.write(
                        "MIST Entropy: %f\n" % (sum(entropy1d) - sum(MIST_list))
                    )
                    f_eps.write(
                        "MIE Entropy (kcal/mol): %f\n"
                        % ((sum(entropy1d) - MI) * conversion_factor)
                    )
                    f_eps.write(
                        "MIST Entropy (kcal/mol): %f\n"
                        % ((sum(entropy1d) - sum(MIST_list)) * conversion_factor)
                    )
                    s_profile.append(
                        (sum(entropy1d) - sum(MIST_list)) * conversion_factor
                    )

                else:
                    bond_entropy = sum(entropy1d[:bond_num]) * (atom_num - 1) / bond_num
                    angle_entropy = (
                        sum(entropy1d[bond_num : bond_num + angle_num])
                        * (atom_num - 2)
                        / angle_num
                    )
                    torsion_entropy = (
                        sum(entropy1d[bond_num + angle_num :])
                        * (atom_num - 3)
                        / torsion_num
                    )
                    bond_mist = (
                        sum(MIST_list[:bond_num]) * (atom_num - 2) / (bond_num - 1)
                    )
                    angle_mist = (
                        sum(MIST_list[bond_num : bond_num + angle_num])
                        * (atom_num - 3)
                        / (angle_num - 1)
                    )
                    torsion_mist = (
                        sum(MIST_list[bond_num + angle_num :])
                        * (atom_num - 4)
                        / (torsion_num - 1)
                    )
                    total = (
                        bond_entropy
                        + angle_entropy
                        + torsion_entropy
                        - bond_mist
                        - angle_mist
                        - torsion_mist
                    )
                    f_eps.write("Bond Entropy: %f\n" % bond_entropy)
                    f_eps.write("Angle Entropy: %f\n" % angle_entropy)
                    f_eps.write("Torsion Entropy: %f\n" % torsion_entropy)
                    f_eps.write("MIST: %f\n" % (bond_mist + angle_mist + torsion_mist))
                    f_eps.write(
                        "MIST Entropy (kcal/mol): %f\n" % (conversion_factor * total)
                    )
                    s_profile.append(conversion_factor * total)

        np.save(os.path.join(eps_output_dir, "eps.npy"), np.array(s_profile))
        with open(os.path.join(eps_output_dir, "eps.log"), "w") as fw:
            fw.write(
                "Ensemble No. \t RC High \t RC Low \t Entropy(kcal/mol) \t Relative(kcal/mol) \t Augmented Snapshots \t Real Snapshots\n"
            )
            for i in range(len(s_profile)):
                if args.rc_direction == "decreasing":
                    low, high = runavg[i] - dx, runavg[i]
                else:
                    low, high = runavg[i], runavg[i] + dx
                fw.write(
                    f"{i+1} \t {high:.4f} \t {low:.4f} \t {s_profile[i]:.4f} \t {(s_profile[i] - s_profile[0]):.4f} \t {sample_size[i]} \t {real_sample_size[i]}\n"
                )
