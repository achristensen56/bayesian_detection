import numpy as np
from scipy import stats
from scipy.stats import truncnorm, norm
import pandas as pd

#TODO make an sklearn style wrapper for RL models, double check SVM learning rule against adams code 

def calcab(myclip_a, myclip_b, loc, scale):
    return (myclip_a - loc) / scale, (myclip_b - loc) / scale

def simple_detection_sim(amplitude_of_tone, 
                         n_trials = 1000, 
                         noise_scale= .3, 
                         noise_mean = .1, 
                         prior_init= .5, 
                         boundary = .5, 
                         n_repeats = 1, 
                         alpha = .1, 
                         boundary_alpha = .1, 
                         stim_prob = None):

    #generate the noise distribution
    a, b = calcab(0, 1, loc = noise_mean, scale = noise_scale)
    noise_distrib = truncnorm(a, b, loc = noise_mean, scale = noise_scale)

    trial_dat = []
    for repeat in range(n_repeats):
        #generate the signal distribution
        #if a tone is present, it is drawn from a uniform distribution between .01 and 1 -- that is interpreted as the amplitude of the tone
        #a negative amplitude in dB would nean a lower amplitude than the mean of the noise distribution
        white_noise = noise_distrib.rvs(n_trials)
        stimulus = amplitude_of_tone + white_noise
        prior_signal = prior_init

        for trial in range(n_trials):
            #need to recalculate the noise distribution as a function of the priors
            #could do this empirically for the signal distribution??
            
            tp = np.random.choice([0,1], size = 10000, p = [1-prior_signal, prior_signal])
            stim = np.random.uniform(.01, 1, size = 10000)*tp
            emperical_distrib = stim + noise_distrib.rvs(10000)

            l_data = norm.pdf(norm.ppf(stats.ecdf(emperical_distrib).cdf.evaluate(stimulus[trial])))
            l_data = np.clip(l_data, 1e-4, 1-1e-4)

            #how surprising is teh data given the known noise and signal distributions
            #surprise = np.clip(((1/(l_data) - 1)**.25), 0, 10)
            surprise = 1/l_data

            l_stimulus_stim = norm.pdf(norm.ppf(stats.ecdf(emperical_distrib[tp == 1]).cdf.evaluate(stimulus[trial])))      
            l_stimulus_stim = np.clip(l_stimulus_stim, 1e-4, 1-1e-4)

    
            emperical_noise_distrib = stats.ecdf(emperical_distrib[tp == 0])
            l_stimulus_noise = norm.pdf(norm.ppf(emperical_noise_distrib.cdf.evaluate(stimulus[trial])))
            l_stimulus_noise = np.clip(l_stimulus_noise, 1e-4, 1-1e-4)
  
            pstim = l_stimulus_stim*prior_signal/(l_stimulus_stim*prior_signal + l_stimulus_noise*(1-prior_signal))
            
            
            pnoise = l_stimulus_noise*(1-prior_signal)/(l_stimulus_stim*prior_signal + l_stimulus_noise*(1-prior_signal))
            

            
            p_tone_given_signal = pstim/(pstim + pnoise)

            detected = p_tone_given_signal > boundary

            if detected:
                confidence = p_tone_given_signal 
                direction = 1
            else:
                confidence = 1-p_tone_given_signal 
                direction = -1   

            
            correct = int(amplitude_of_tone[trial] > 0) == int(detected)
            sound_present = amplitude_of_tone[trial] > 0


            prior_signal +=  alpha*surprise*(confidence)*direction
            prior_signal = np.clip(prior_signal, 0.01, .99)

            #update the detection threshold based on reward prediction error to maximize reward
            
  
            trial_dict = {'stimulus': amplitude_of_tone[trial], 
                          'internal_stim': stimulus[trial],
                          'white_noise': white_noise[trial],
                          'prior_signal': prior_signal,
                          'p_signal': l_stimulus_stim,
                          'p_noise': l_stimulus_noise, 
                          'p_tone_given_signal': p_tone_given_signal,
                          'choice': detected, 
                          'trial': trial, 
                          'correct': correct, 
                          'sound_present': sound_present,
                          'confidence': confidence, 
                          'l_data': l_data,
                          'spe': surprise,
                          'update': alpha*surprise*(confidence*direction),
                          'surprise': surprise,
                          'boundary': boundary, 
                          'prior': stim_prob[trial]}
            trial_dat.append(pd.DataFrame(trial_dict, index = [trial]))

    return pd.concat(trial_dat)


def bayesian_detection(amplitude_of_tone, 
                         noise_trials,
                         scale = .3,
                         n_trials = 1000,
                         prior_init= .5, 
                         boundary = .5, 
                         n_repeats = 1, 
                         alpha = .1, 
                         stim_prob = None):

    #generate the noise distribution

    noise_distrib = norm(loc = 0, scale = scale)
    trial_dat = []


    for repeat in range(n_repeats):
        #generate the signal distribution
        white_noise = noise_distrib.rvs(n_trials)
        stimulus = amplitude_of_tone + white_noise #there is perceptual noise on every trial
        prior_signal = prior_init

        for trial in range(n_trials):
            #need to recalculate the noise distribution as a function of the priors
            #could do this empirically for the signal distribution??
            tp = np.random.choice([0,1], size = 10000, p = [1-prior_signal, prior_signal])

            stim = np.random.choice(stimulus[noise_trials == 0], size = 10000)*tp
            noise = np.random.choice(stimulus[noise_trials == 1], size = 10000)*(tp == 0)
            emperical_distrib  = stim + noise
            l_data = norm.pdf(norm.ppf(stats.ecdf(emperical_distrib).cdf.evaluate(stimulus[trial])))
            l_data = np.clip(l_data, 1e-4, 1-1e-4)

            #how surprising is teh data given the known noise and signal distributions
            #surprise = np.clip(((1/(l_data) - 1)**.25), 0, 10)
            surprise = -np.log(l_data)

            l_stimulus_stim = norm.pdf(norm.ppf(stats.ecdf(emperical_distrib[tp == 1]).cdf.evaluate(stimulus[trial])))      
            l_stimulus_stim = np.clip(l_stimulus_stim, 1e-4, 1-1e-4)

    
            emperical_noise_distrib = stats.ecdf(emperical_distrib[tp == 0])
            l_stimulus_noise = norm.pdf(norm.ppf(emperical_noise_distrib.cdf.evaluate(stimulus[trial])))
            l_stimulus_noise = np.clip(l_stimulus_noise, 1e-4, 1-1e-4)
  
            pstim = l_stimulus_stim*prior_signal/(l_stimulus_stim*prior_signal + l_stimulus_noise*(1-prior_signal))
            
            
            pnoise = l_stimulus_noise*(1-prior_signal)/(l_stimulus_stim*prior_signal + l_stimulus_noise*(1-prior_signal))
            
            SPE  =   np.log(pstim/pnoise) - np.log(prior_signal/(1-prior_signal))
            #surprise = np.abs(SPE)

            
            p_tone_given_signal = pstim/(pstim + pnoise)

            detected = p_tone_given_signal > boundary

            if detected:
                confidence = p_tone_given_signal 
                direction = 1
            else:
                confidence = 1-p_tone_given_signal 
                direction = -1   

            
            correct = int(amplitude_of_tone[trial] > 0) == int(detected)
            sound_present = amplitude_of_tone[trial] > 0


            prior_signal +=  alpha*(surprise)*direction
            prior_signal = np.clip(prior_signal, 0.01, .99)

            #update the detection threshold based on reward prediction error to maximize reward
            
  
            trial_dict = {'stimulus': amplitude_of_tone[trial], 
                          'internal_stim': stimulus[trial],
                          'white_noise': white_noise[trial],
                          'prior_signal': prior_signal,
                          'p_signal': l_stimulus_stim,
                          'p_noise': l_stimulus_noise, 
                          'p_tone_given_signal': p_tone_given_signal,
                          'choice': detected, 
                          'trial': trial, 
                          'correct': correct, 
                          'sound_present': sound_present,
                          'confidence': confidence, 
                          'l_data': l_data,
                          'spe': SPE,
                          'update': alpha*surprise*(confidence*direction),
                          'surprise': surprise,
                          'boundary': boundary, 
                          'prior': stim_prob[trial]}
            trial_dat.append(pd.DataFrame(trial_dict, index = [trial]))

    return pd.concat(trial_dat)

def simple_discrimination_sim(DV,
                              n_trials = 1000, 
                              noise_scale = .3,
                              noise_mean = 0, 
                              prior_init = .5, 
                              boundary_init = 0,
                              n_repeats = 1,
                              prior_alpha = 0, 
                              boundary_alpha = .1, 
                              value_alpha = .1, 
                              value_init = [-1, 1], 
                              stimulus_prob = None):
    
        #generate the noise distribution
    #a, b = calcab(-1, 1, loc = noise_mean, scale = noise_scale)
    noise_distrib = norm(loc = noise_mean, scale = noise_scale)

    #want to find the 

    trial_dat = []
    for repeat in range(n_repeats):
        perceptual_noise = noise_distrib.rvs(n_trials)
        prior_left = prior_init
        stimulus = perceptual_noise + DV.copy()
        boundary = boundary_init
        value = value_init

        for trial in range(n_trials):
            p_left = norm.cdf(boundary, loc = stimulus[trial], scale = noise_scale)
            p_right = 1 - p_left
            
            p_left = p_left*prior_left/(p_left*prior_left + p_right*(1-prior_left))
            p_right = 1 - p_left

            choice = [0, 1][np.argmax([p_left*value[0], p_right*value[1]])]
            
            #prior update
            prior_left = (1-prior_alpha)*prior_left + prior_alpha*p_left

            chosen_dir = [-1, 1][choice]
            p_chosen = [p_left, p_right][choice]
            v_chosen = value[choice]

            correct_dir = [-1, 1][int(DV[trial] > 0)]
            correct = int(chosen_dir == correct_dir)

            #boundary update
            boundary_update = abs(stimulus[trial] - boundary)*correct_dir
            boundary -= boundary_alpha*boundary_update 

            #value update
            RPE = correct - v_chosen*p_chosen 
            value[choice] += value_alpha*RPE



            trial_dict = {'stimulus': DV[trial], 
                          'noise': perceptual_noise[trial], 
                          'prior_left': prior_left,
                          'correct': correct,
                          'RPE': RPE,
                          'answer': correct_dir,
                          'sim_number': repeat,
                          'p_chosen': p_chosen,
                          'p_left': p_left, 
                          'p_right': 1-p_left, 
                          'choice': choice, 
                          'boundary': boundary,
                          'trial': trial, 
                          'prior': stimulus_prob[trial] }

            trial_dat.append(pd.DataFrame(trial_dict, index = [trial]))
    return pd.concat(trial_dat) 

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

#define a helper function to generate stimulus data with prior probabilities in blocks
def generate_trials(block_len, n_blocks, right_probs, kind = "discrimination"):

    if kind == "Discrimination":
        stim = []
        p_right = []
        for i in range(n_blocks):
            pr = np.random.choice(right_probs)
            for j in range(block_len):
                x_i = np.random.uniform(0, 1)*np.random.choice(np.array([-1, 1.0]), p = [1-pr, pr])
                stim.append(x_i)
                p_right.append(pr)
        return np.array(stim), np.array(p_right), np.array(np.sign(stim))

    if kind == "Detection":
     #   tone_present = np.random.choice([0,1], p = [.5, .5], size = n_trials)
     #   amplitude_of_tone = np.random.uniform(.01, 1, size = n_trials)
        stim = []
        p_right = []
        tone_present = []
        for i in range(n_blocks):
            pr = np.random.choice(right_probs)
            for j in range(block_len):
                tp = np.random.choice([0,1], p = [1-pr, pr])
                x_i = np.random.uniform(.01, 1)*tp
                tone_present.append(tp)
                stim.append(x_i)
                p_right.append(pr)

        return np.array(stim), np.array(p_right), np.array(tone_present)
