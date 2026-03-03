import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simpy
import warnings
from scipy.stats import t
from sim_tools.distributions \
    import Exponential, Lognormal, GroupedContinuousEmpirical

class CritCarePt:
    """
    Represents a patient in the system.

    Attributes
    ----------
    id : int
        Unique patient identifier.
    type: str
        The type of patient
    queue_time : float
        Time the patient spent waiting for a bed in hours.
    time_in_ccu : float
        Actual time the patient spent occupying a bed during the simulation.
    """
    def __init__(self, p_id, p_type):
        """
        Initialise a new patient.
        """
        self.id = p_id
        self.type = p_type
        self.queue_time = 0
        self.time_in_ccu = 0

class CritCareUnit:
    """
    Discrete event simulation of the system.

    Attributes
    ----------
    random_seed : int
        Seed used to reproduce a specific run.
    unplanned_params_dict : dict
        Dictionary of distribution parameters for each patient type.
    time_to_clean : float
        The time taken for intensive cleaning of a bed in hours.
    num_beds : int
        Number of CCU beds available.
    sim_duration : float
        Duration of the simulation after the warm-up period.
    warm_up : float
        Warm-up period duration in hours
        (results collected only after this time).
    unplanned_dist_params : dict
        Dictionary of parameters for inter-arrival and length-of-stay
        distributions for different unplanned patient types.
    elective_params : dict
        Dictionary of parameters for the inter-arrival and length-of-stay
        distributions for elective surgery patients.
    first_patient_post_warm_up : int
        The identifier of the first patient to be included in results
        calculations following the warm-up period.
    env : simpy.Environment
        The SimPy simulation environment.
    patient_counter : int
        Running count of patients who have arrived.
    num_cancellations : int
        Running count of the number of elective surgery patients who have had
        their operations cancelled.
    sim_admitted_pts : list of CritCarePt
        Patient objects created after the warm-up period.
    bed : simpy.Resource
        SimPy resource representing the pool of beds.
    results_df : pandas.DataFrame
        Per-patient results (Queue Time, Length of Stay).
    sampled_inter_arrival_times : dict
        The sampled interrarrival times during the simulation.
    sampled_stay_durations : dict
        The sampled lengths of stay during the simulation.
    mean_queue_time : float
        Average queue time across all post-warm-up patients.
    bed_utilisation : float
        Average bed utilisation as a percentage.
    bed_occupancy : float
        Average number of beds occupied.
    pt_inter_arrival_dist : dict
        Dictionary of inter-arrival distributions for different patient types.
    pt_len_stay_dist : dict
        Dictionary of length-of-stay distributions for different patient types.
    """
    def __init__(self, random_seed, unplanned_params_dict, elective_dict,
                 time_to_clean, number_of_beds, simulation_duration,
                 warm_up_period):
        """
        Initialise the simulation model.
        """
        self.random_seed = random_seed
        self.unplanned_dist_params = unplanned_params_dict
        self.elective_params = elective_dict
        self.clean_time = time_to_clean
        self.num_beds = number_of_beds
        self.sim_duration = simulation_duration
        self.warm_up = warm_up_period
        self.pt_inter_arrival_dist = {}
        self.pt_len_stay_dist = {}
        self.env = simpy.Environment()
        self.patient_counter = 0
        self.num_cancellations = 0
        self.first_patient_post_warm_up = 0
        self.sim_admitted_pts = []
        self.queue_for_beds = []
        self.bed = simpy.Resource(self.env, capacity=self.num_beds)
        self.results_df = pd.DataFrame()
        self.results_df["Patient ID"] = [1]
        self.results_df["Patient Type"] = [" "]
        self.results_df["Queue Time"] = [0.0]
        self.results_df["Length of Stay"] = [0.0]
        self.results_df.set_index("Patient ID", inplace=True)
        self.sampled_inter_arrival_times = {}
        self.sampled_stay_durations = {}
        self.mean_queue_time = 0.0
        self.bed_utilisation = 0.0
        self.bed_occupancy = 0.0
        self.initialise_distributions()

    def initialise_distributions(self):
        """
        Initialise patient distributions to sample from.
        """
        unplanned_patient_types = list(self.unplanned_dist_params.keys())
        ss = np.random.SeedSequence(self.random_seed)
        seeds = ss.spawn(2 * len(unplanned_patient_types) + 2)
        self.pt_inter_arrival_dist = {}
        self.pt_len_stay_dist = {}
        for i in range(len(unplanned_patient_types)):
            self.pt_inter_arrival_dist[
                unplanned_patient_types[i]] = Exponential(
                mean=self.unplanned_dist_params[
                    unplanned_patient_types[i]]["Mean Inter-Arrival Time"],
                random_seed=seeds[i]
            )
            self.pt_len_stay_dist[
                unplanned_patient_types[i]] = Lognormal(
                mean=self.unplanned_dist_params[unplanned_patient_types[i]][
                    "Mean Length of Stay"],
                stdev=self.unplanned_dist_params[unplanned_patient_types[i]][
                    "Length of Stay Standard Deviation"],
                random_seed=seeds[i + len(unplanned_patient_types)]
            )
            self.sampled_inter_arrival_times[unplanned_patient_types[i]] = []
            self.sampled_stay_durations[unplanned_patient_types[i]] = []
        self.pt_inter_arrival_dist[
            "Elective Surgery"] = GroupedContinuousEmpirical(
            lower_bounds=self.elective_params["Lower Bounds"],
            upper_bounds=self.elective_params["Upper Bounds"],
            freq=self.elective_params["Frequencies"],
            random_seed=seeds[len(unplanned_patient_types)]
        )
        self.pt_len_stay_dist["Elective Surgery"] = Lognormal(
            mean=self.elective_params["Mean Length of Stay"],
            stdev=self.elective_params[
                "Length of Stay Standard Deviation"],
            random_seed=seeds[len(unplanned_patient_types) + 1]
        )
        self.sampled_inter_arrival_times["Elective Surgery"] = []
        self.sampled_stay_durations["Elective Surgery"] = []

    def generate_patient_referrals(self, patient_type):
        """
        Generate arriving patients.

        Each iteration increments the patient counter, spawns an admission
        process for the new patient, then waits for a
        sampled inter-arrival time before looping.
        """
        while True:
            self.patient_counter += 1
            p = CritCarePt(self.patient_counter, patient_type)
            self.env.process(self.admit_unit(p))
            sampled_inter = self.pt_inter_arrival_dist[patient_type].sample()
            if self.env.now > self.warm_up:
                self.sampled_inter_arrival_times[patient_type].append(
                    sampled_inter)
            yield self.env.timeout(sampled_inter)

    def admit_unit(self, patient):
        """
        Models a single patient's admission.

        The patient queues for a bed, occupies it for a sampled length of
        stay, and then releases it.  Results are only recorded after the
        warm-up period has elapsed.
        """
        if len(self.queue_for_beds) > 0 and patient.type == "Elective Surgery":
            if self.env.now > self.warm_up:
                self.num_cancellations += 1
                self.results_df.at[patient.id, "Patient Type"] = (
                        "Elective Surgery")
                self.results_df.at[patient.id, "Queue Time"] = (
                        np.nan)
                self.results_df.at[patient.id, "Length of Stay"] = (
                        np.nan)
        else:
            start_q_bed = self.env.now
            self.queue_for_beds.append(patient)
            with self.bed.request() as req:
                yield req
                self.queue_for_beds.remove(patient)
                end_q_bed = self.env.now
                patient.queue_time = end_q_bed - start_q_bed
                sampled_len_stay = self.pt_len_stay_dist[patient.type].sample()
                sim_time_remain = self.sim_duration + self.warm_up \
                                  - self.env.now
                len_stay_before_end_sim = min(sampled_len_stay,
                                              sim_time_remain)
                patient.time_in_ccu = len_stay_before_end_sim
                if self.env.now > self.warm_up:
                    self.results_df.at[patient.id, "Patient Type"] = (
                        patient.type)
                    self.results_df.at[patient.id, "Queue Time"] = (
                        patient.queue_time)
                    self.results_df.at[patient.id, "Length of Stay"] = (
                        sampled_len_stay)
                    self.sim_admitted_pts.append(patient)
                yield self.env.timeout(sampled_len_stay + self.clean_time)

    def audit_bed_utilisation(self):
        """
        Calculate bed utilisation as a percentage.

        Utilisation is computed as the total occupied bed-time divided by
        the total available bed-time (beds × simulation duration), expressed
        as a percentage.
        """
        treatment_durations = [i.time_in_ccu for i in self.sim_admitted_pts]
        return (sum(treatment_durations) * 100
                / (self.num_beds * self.sim_duration))

    def calculate_run_results(self):
        """
        Aggregate per-run summary statistics.

        Removes the placeholder row from results_df, then computes
        mean queue time, mean length of stay, and bed utilisation for the
        run.
        """
        self.results_df.drop([1], inplace=True)
        self.first_patient_post_warm_up = min(self.results_df.index)
        self.mean_queue_time = self.results_df["Queue Time"].mean()
        self.bed_utilisation = self.audit_bed_utilisation()
        self.bed_occupancy = self.bed_utilisation * self.num_beds / 100

    def run(self):
        """
        Execute the simulation.

        Registers the patient-arrival generator with the SimPy environment,
        runs the simulation for warm_up + sim_duration hours, then
        calculates summary results.
        """
        patient_types = list(self.pt_inter_arrival_dist.keys())
        for pt_type in patient_types:
            self.env.process(self.generate_patient_referrals(pt_type))
        self.env.run(until=(self.sim_duration + self.warm_up))
        self.calculate_run_results()

class ScenarioTrial:
    """
    Runs multiple replications of a simulation of the CCU system for a fixed
    scenario and tracks convergence via confidence intervals. The minimum
    number of replications required to reach a desired precision in both the
    mean queue time and bed utilisation metrics is also determined.

    Attributes
    ----------
    num_reps : int
        Number of simulation replications to run.
    unplanned_params_dict : dict
        Dictionary of distribution parameters for each patient type.
    time_to_clean : float
        The time taken for intensive cleaning of a bed in hours.
    num_beds : int
        Number of beds passed to each replication.
    sim_duration : float
        Simulation duration per replication in hours.
    warm_up : float
        Warm-up period per replication in hours.
    unplanned_dist_params : dict
        Dictionary of parameters for inter-arrival and length-of-stay
        distributions for different unplanned patient types.
    elective_params : dict
        Dictionary of parameters for the inter-arrival and length-of-stay
        distributions for elective surgery patients.
    alpha : float
        Significance level for confidence intervals (default 0.05).
    precision : float
        Desired relative precision as a proportion (default 0.05 = 5%).
    ci_fig_size : tuple
        Figure size used by the confidence-interval plot methods.
    min_reps_wait : int
        Minimum replications to reach precision for mean queue time
        (-1 if not reached).
    min_reps_util : int
        Minimum replications to reach precision for bed utilisation
        (-1 if not reached).
    min_reps : int
        Overall minimum replications required (-1 if not reached).
    run_results_dict : dict
        Maps replication labels to their per-patient results DataFrames.
    trial_results_df : pandas.DataFrame
        One row per replication with mean queue time, bed utilisation,
        and occupancy.
    overall_results : pandas.DataFrame
        The overall results of the trial.
    var_wait_df : pandas.DataFrame
        Running statistics (mean, CI bounds, % deviation) for queue time.
    var_util_df : pandas.DataFrame
        Running statistics for bed utilisation.
    var_beds_df : pandas.DataFrame
        Running statistics for bed occupancy.
    var_cancels_df : pandas.DataFrame
        Running statistics for cancellations.
    """
    def __init__(self, number_of_reps, unplanned_params_dict, elective_dict,
                 time_to_clean, number_of_beds, simulation_duration,
                 warm_up_period, ci_alpha=0.05, desired_precision=0.05):
        self.num_reps = number_of_reps
        self.unplanned_dist_params = unplanned_params_dict
        self.elective_params = elective_dict
        self.clean_time = time_to_clean
        self.num_beds = number_of_beds
        self.sim_duration = simulation_duration
        self.warm_up = warm_up_period
        self.alpha = ci_alpha
        self.precision = desired_precision
        self.min_reps_wait = 0
        self.min_reps_util = 0
        self.min_reps_beds = 0
        self.min_reps = 0
        self.run_results_dict = {}
        self.trial_results_df = pd.DataFrame()
        self.trial_results_df["Mean Wait for a Bed"] = [0.0]
        self.trial_results_df["Bed Utilisation"] = [0.0]
        self.trial_results_df["Bed Occupancy"] = [0.0]
        self.trial_results_df["Number of Cancellations"] = [0]
        self.overall_results_df = pd.DataFrame()
        self.var_wait_df = pd.DataFrame()
        self.var_util_df = pd.DataFrame()
        self.var_beds_df = pd.DataFrame()

    def run_trial(self):
        """
        Execute all replications, populate trial_results_df, and compute
        running confidence intervals.

        For each replication a new CritCareUnit object is instantiated,
        run, and its results stored.  The index of trial_results_df is
        set to 1-based replication numbers.

        After running every replication, cumulative means, running variances,
        standard errors, and confidence intervals half-widths for mean queue
        time, bed utilisation, and bed occupancy are computed.

        The min_reps_wait, min_reps_util, min_reps_beds and min_reps attributes
        are set to the first replication index at which the relative deviation
        falls below the desired precision. A warning is issued for any metric
        that never reaches the desired precision.
        """
        for run in range(self.num_reps):
            ccu_model = CritCareUnit(
                random_seed=run,
                unplanned_params_dict=self.unplanned_dist_params,
                elective_dict=self.elective_params,
                time_to_clean=self.clean_time,
                number_of_beds=self.num_beds,
                simulation_duration=self.sim_duration,
                warm_up_period=self.warm_up
            )
            ccu_model.run()
            run_results = ccu_model.results_df
            self.run_results_dict[f"Replication {run+1}"] = run_results
            self.trial_results_df.loc[run, "Mean Wait for a Bed"] \
                = [ccu_model.mean_queue_time]
            self.trial_results_df.loc[run, "Bed Utilisation"] \
                = [ccu_model.bed_utilisation]
            self.trial_results_df.loc[run, "Bed Occupancy"] \
                = [ccu_model.bed_occupancy]
            self.trial_results_df.loc[run, "Number of Cancellations"] \
                = [ccu_model.num_cancellations]
        self.trial_results_df.index = np.arange(1, self.num_reps + 1)
        self.trial_results_df.index.name = "Replication"

        mean_wait = self.trial_results_df["Mean Wait for a Bed"].mean()
        std_wait = self.trial_results_df["Mean Wait for a Bed"].std()
        mean_util = self.trial_results_df["Bed Utilisation"].mean()
        std_util = self.trial_results_df["Bed Utilisation"].std()
        mean_beds = self.trial_results_df["Bed Occupancy"].mean()
        std_beds = self.trial_results_df["Bed Occupancy"].std()
        mean_cancels = self.trial_results_df["Number of Cancellations"].mean()
        std_cancels = self.trial_results_df["Number of Cancellations"].std()

        self.overall_results_df = pd.DataFrame({
            "Mean": [mean_wait, mean_util, mean_beds, mean_cancels],
            "Standard Deviation": [std_wait, std_util, std_beds, std_cancels]
        })

        self.overall_results_df.index = ["Mean Wait for a Bed",
                                         "Bed Utilisation",
                                         "Bed Occupancy",
                                         "Number of Cancellations"]

        degrees_freedom = self.num_reps - 1
        t_value = t.ppf(1 - (self.alpha / 2),  degrees_freedom)

        mean_wait_list = self.trial_results_df["Mean Wait for a Bed"].tolist()
        cumulative_mean_wait = [mean_wait_list[0]]
        running_var_wait = [0.0]
        for i in range(1, self.num_reps):
            cumulative_mean_wait.append(cumulative_mean_wait[i - 1]
                                        + (mean_wait_list[i]
                                           - cumulative_mean_wait[i - 1])
                                        / (i + 1))
            running_var_wait.append(running_var_wait[i - 1]
                                    + (mean_wait_list[i]
                                       - cumulative_mean_wait[i - 1])
                                    * (mean_wait_list[i]
                                       - cumulative_mean_wait[i]))
        with np.errstate(divide='ignore', invalid='ignore'):
            running_std_wait = np.sqrt(running_var_wait
                                       / np.arange(self.num_reps))
        with np.errstate(divide='ignore', invalid='ignore'):
            std_error_wait = running_std_wait / np.sqrt(np.arange(1,
                                                                  self.num_reps
                                                                  + 1))
        half_width_wait = t_value * std_error_wait
        upper_wait = cumulative_mean_wait + half_width_wait
        lower_wait = cumulative_mean_wait - half_width_wait
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation_wait = (half_width_wait / cumulative_mean_wait) * 100
        self.var_wait_df = pd.DataFrame([mean_wait_list, cumulative_mean_wait,
                                         running_std_wait,
                                         lower_wait, upper_wait,
                                         deviation_wait]).T
        self.var_wait_df.columns = ["Mean", "Cumulative Mean",
                                    "Standard Deviation", "Lower Interval",
                                    "Upper Interval", "% Deviation"]
        self.var_wait_df.index = np.arange(1, self.num_reps + 1)
        self.var_wait_df.index.name = "Replications"

        wait_reps_below_precision = self.var_wait_df.loc[
            self.var_wait_df["% Deviation"]
            <= self.precision*100].index.tolist()
        if len(wait_reps_below_precision) == 0:
            message = "WARNING: The replications do not reach the desired " \
                + "precision for mean wait for a bed."
            warnings.warn(message)
            self.min_reps_wait = -1
        else:
            self.min_reps_wait = wait_reps_below_precision[0]

        mean_util_list = self.trial_results_df[
            "Bed Utilisation"].tolist()
        cumulative_mean_util = [mean_util_list[0]]
        running_var_util = [0.0]
        for i in range(1, self.num_reps):
            cumulative_mean_util.append(cumulative_mean_util[i - 1]
                                        + (mean_util_list[i]
                                           - cumulative_mean_util[i - 1])
                                        / (i+1))
            running_var_util.append(running_var_util[i - 1]
                                    + (mean_util_list[i]
                                       - cumulative_mean_util[i - 1])
                                    * (mean_util_list[i]
                                       - cumulative_mean_util[i]))
        with np.errstate(divide="ignore", invalid="ignore"):
            running_std_util = np.sqrt(running_var_util
                                       / np.arange(self.num_reps))
        with np.errstate(divide="ignore", invalid="ignore"):
            std_error_util = running_std_util / np.sqrt(np.arange(1,
                                                                  self.num_reps
                                                                  + 1))
        half_width_util = t_value * std_error_util
        upper_util = cumulative_mean_util + half_width_util
        lower_util = cumulative_mean_util - half_width_util
        with np.errstate(divide="ignore", invalid="ignore"):
            deviation_util = (half_width_util / cumulative_mean_util) * 100
        self.var_util_df = pd.DataFrame([mean_util_list, cumulative_mean_util,
                                         running_std_util,
                                         lower_util, upper_util,
                                         deviation_util]).T
        self.var_util_df.columns = ["Mean", "Cumulative Mean",
                                    "Standard Deviation", "Lower Interval",
                                    "Upper Interval", "% Deviation"]
        self.var_util_df.index = np.arange(1, self.num_reps+1)
        self.var_util_df.index.name = "Replications"

        util_reps_below_precision = self.var_util_df.loc[
            self.var_util_df["% Deviation"]
            <= self.precision * 100].index.tolist()
        if len(util_reps_below_precision) == 0:
            message = "WARNING: The replications do not reach the desired " \
                + "precision for bed utilisation."
            warnings.warn(message)
            self.min_reps_util = -1
        else:
            self.min_reps_util = util_reps_below_precision[0]

        mean_beds_list = self.trial_results_df[
            "Bed Occupancy"].tolist()
        cumulative_mean_beds = [mean_beds_list[0]]
        running_var_beds = [0.0]
        for i in range(1, self.num_reps):
            cumulative_mean_beds.append(cumulative_mean_beds[i - 1]
                                        + (mean_beds_list[i]
                                           - cumulative_mean_beds[i - 1])
                                        / (i + 1))
            running_var_beds.append(running_var_beds[i - 1]
                                    + (mean_beds_list[i]
                                       - cumulative_mean_beds[i - 1])
                                    * (mean_beds_list[i]
                                       - cumulative_mean_beds[i]))
        with np.errstate(divide='ignore', invalid='ignore'):
            running_std_beds = np.sqrt(running_var_beds
                                       / np.arange(self.num_reps))
        with np.errstate(divide='ignore', invalid='ignore'):
            std_error_beds = running_std_beds / np.sqrt(np.arange(1,
                                                                  self.num_reps
                                                                  + 1))
        half_width_beds = t_value * std_error_beds
        upper_beds = cumulative_mean_beds + half_width_beds
        lower_beds = cumulative_mean_beds - half_width_beds
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation_beds = (half_width_beds / cumulative_mean_beds) * 100
        self.var_beds_df = pd.DataFrame([mean_beds_list, cumulative_mean_beds,
                                         running_std_beds,
                                         lower_beds, upper_beds,
                                         deviation_beds]).T
        self.var_beds_df.columns = ["Mean", "Cumulative Mean",
                                    "Standard Deviation", "Lower Interval",
                                    "Upper Interval", "% Deviation"]
        self.var_beds_df.index = np.arange(1, self.num_reps + 1)
        self.var_beds_df.index.name = "Replications"

        beds_reps_below_precision = self.var_wait_df.loc[
            self.var_beds_df["% Deviation"]
            <= self.precision*100].index.tolist()
        if len(beds_reps_below_precision) == 0:
            message = "WARNING: The replications do not reach the desired " \
                + "precision for bed occupancy."
            warnings.warn(message)
            self.min_reps_beds = -1
        else:
            self.min_reps_beds = beds_reps_below_precision[0]

        mean_cancels_list = self.trial_results_df[
            "Number of Cancellations"].tolist()
        cumulative_mean_cancels = [mean_cancels_list[0]]
        running_var_cancels = [0.0]
        for i in range(1, self.num_reps):
            cumulative_mean_cancels.append(cumulative_mean_cancels[i - 1]
                                           + (mean_cancels_list[i]
                                           - cumulative_mean_cancels[i - 1])
                                           / (i + 1))
            running_var_cancels.append(running_var_cancels[i - 1]
                                       + (mean_cancels_list[i]
                                       - cumulative_mean_cancels[i - 1])
                                       * (mean_cancels_list[i]
                                       - cumulative_mean_cancels[i]))
        with np.errstate(divide='ignore', invalid='ignore'):
            running_std_cancels = np.sqrt(running_var_cancels
                                          / np.arange(self.num_reps))
        with np.errstate(divide='ignore', invalid='ignore'):
            std_error_cancels = running_std_cancels / np.sqrt(np.arange(1,
                self.num_reps + 1))
        half_width_cancels = t_value * std_error_cancels
        upper_cancels = cumulative_mean_cancels + half_width_cancels
        lower_cancels = cumulative_mean_cancels - half_width_cancels
        with np.errstate(divide='ignore', invalid='ignore'):
            deviation_cancels = (half_width_cancels
                                 / cumulative_mean_cancels) * 100
        self.var_cancels_df = pd.DataFrame([mean_cancels_list,
                                            cumulative_mean_cancels,
                                            running_std_cancels,
                                            lower_cancels, upper_cancels,
                                            deviation_cancels]).T
        self.var_cancels_df.columns = ["Mean", "Cumulative Mean",
                                       "Standard Deviation", "Lower Interval",
                                       "Upper Interval", "% Deviation"]
        self.var_cancels_df.index = np.arange(1, self.num_reps + 1)
        self.var_cancels_df.index.name = "Replications"

        cancels_reps_below_precision = self.var_cancels_df.loc[
            self.var_cancels_df["% Deviation"]
            <= self.precision*100].index.tolist()
        if len(cancels_reps_below_precision) == 0:
            message = "WARNING: The replications do not reach the desired " \
                + "precision for mean number of operations cancelled."
            warnings.warn(message)
            self.min_reps_cancels = -1
        else:
            self.min_reps_cancels = cancels_reps_below_precision[0]

        if (self.min_reps_wait > 0
            and self.min_reps_util > 0
            and self.min_reps_cancels > 0):
                self.min_reps = max(
                    self.min_reps_wait,
                    self.min_reps_util,
                    self.min_reps_beds,
                    self.min_reps_cancels
                )
        else:
            self.min_reps = -1

    def plot_reps(self):
        """
        Plot cumulative mean and confidence intervals for all metrics.

        A vertical red dashed line marks the replication at which the
        desired precision is first achieved (if ever).
        """
        fig, axs = plt.subplots(4, figsize=(12, 16), sharex=True)

        self.var_wait_df[
            ["Cumulative Mean", "Lower Interval", "Upper Interval"]
        ].plot(ax=axs[0], legend=False)
        axs[0].grid(ls='--')
        axs[0].set_ylabel("Mean wait for a bed (hours)")
        if self.min_reps > 0:
            axs[0].axvline(x=self.min_reps_wait, ls='--', color='red')

        self.var_util_df[
            ["Cumulative Mean", "Lower Interval", "Upper Interval"]
        ].plot(ax=axs[1], legend=False)
        axs[1].grid(ls='--')
        axs[1].set_ylabel("Bed utilisation (%)")
        if self.min_reps > 0:
            axs[1].axvline(x=self.min_reps_wait, ls='--', color='red')

        self.var_beds_df[
            ["Cumulative Mean", "Lower Interval", "Upper Interval"]
        ].plot(ax=axs[2], legend=False)
        axs[2].grid(ls='--')
        axs[2].set_ylabel("Bed occupancy")
        if self.min_reps > 0:
            axs[2].axvline(x=self.min_reps_wait, ls='--', color='red')

        self.var_cancels_df[
            ["Cumulative Mean", "Lower Interval", "Upper Interval"]
        ].plot(ax=axs[3], legend=False)
        axs[3].grid(ls='--')
        axs[3].set_ylabel("Number of cancellations")
        if self.min_reps > 0:
            axs[3].axvline(x=self.min_reps_wait, ls='--', color='red')

        plt.setp(axs[0].get_xticklabels(), visible=False)
        plt.setp(axs[1].get_xticklabels(), visible=False)
        plt.setp(axs[2].get_xticklabels(), visible=False)
        axs[3].set_xlabel("Replications")

        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=4, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout()

        return axs

class Experiment:
    """
    Runs multiple replications of different scenarios.

    Attributes
    ----------
    scenario_params : dict
        Dictionary of dictionaries containing scenario parameters.
    trial_objects : dict
        Instances of ScenarioTrial for each different scenario.
    """
    def __init__(self, dictionary_of_scenarios):
        self.scenario_params = dictionary_of_scenarios
        self.trial_objects = {}

    def run_experiment(self):
        for scenario in self.scenario_params:
            variables = self.scenario_params[scenario]
            scenario_trial = ScenarioTrial(
                number_of_reps=variables["Number of Replications"],
                unplanned_params_dict=variables[
                    "Unplanned Distribution Parameters Dictionary"
                    ],
                elective_dict=variables[
                    "Planned Admission Distribution Parameters"
                    ],
                time_to_clean=variables["Time to Clean"],
                number_of_beds=variables["Number of Beds"],
                simulation_duration=variables["Simulation Duration"],
                warm_up_period=variables["Warm-Up Period"],
                desired_precision=variables["Desired Precision"]
            )
            scenario_trial.run_trial()
            self.trial_objects[scenario] = scenario_trial