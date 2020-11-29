

#
# wEE=10; wEI=-12; wIE=15; wII=-18
# nE = 3*160; nI = 3*40
# n_neurons = nE + nI
# w_var = 2e-4
# # weights = inh_exc_fully_connected(wEE, wEI, wIE, wII, nE, nI)
# # weights = inh_exc_gauss_fully_connected(wEE, wEI, wIE, wII, nE, nI, w_var)
# weights = inh_exc_gauss_binomial(wEE, wEI, wIE, wII, nE, nI, coupling_probab, w_var)
#
# A_exc = pop_actvity(t, spike_trains[1:nE,:], 2e-3)
# A_inh = pop_actvity(t, spike_trains[nE+1:end,:], 2e-3)
# figure()
# plot(t, A_exc, label="Excitatory")
# plot(t, A_inh, label="Inhibitory")
# legend()

#
# A_filt  = filtfilt(digitalfilter(Lowpass(0.008), Butterworth(7)), A)
# plot(t, A_filt)


# cluster the inhibitory and excitatory populations
# inh_exc_cluster
# coarse_grain(inh_exc_cluster)
