# Ideas backlog (explore after the core H1-H3 run)

Ordered roughly by expected value / effort.

1. **2x phase relative to 1x as misalignment signature.** Coherent spectra
   are complex; we currently only use magnitude. arg(Z_2x) - 2*arg(Z_1x) is
   speed-invariant and textbook-distinct for parallel vs angular
   misalignment. Free feature, columns already computable from the blocked
   spectra. Likely the single cheapest accuracy gain.

2. **Multi-metronome LCM.** Rerun blocks with M in {3, 5, 7} revs and
   intersect: a component coherent at every M is truly shaft-locked; one
   coherent only at M=5 is a 1/5-subharmonic artifact. Cheap disambiguator,
   also directly Shor-flavored (choosing coprime moduli).

3. **Bearing slip estimation as a feature.** For each unsnapped peak, fit
   drift of block-to-block phase: d(phi)/d(block) = 2*pi*slip*order*revs.
   Slip magnitude and its stability separate bearing tones from structural
   resonances (both unsnapped, different physics).

4. **CF on peak RATIOS for unknown kinematics.** Snap o_i/o_j across peak
   pairs: recovers gear tooth ratios and bearing geometry without a
   datasheet. On MAFAULDA (no gearbox) validate on BPFI/BPFO = 5.002/2.998;
   real payoff is field data at NanoPrecise.

5. **Envelope-then-coherent hybrid.** Envelope demodulate a high band
   first, THEN angular resample and coherence-split the envelope. Bearing
   modulation sidebands become low-order components; their coherence ratio
   vs cage order is a severity staging signal.

6. **Pseudo-tacho from MAVEN.** Replace f_nom with the cepstrum-cascade
   RPM estimate (0.14% median error) as a fifth variant "maven": the exact
   production configuration for MachineDoctor. Expectation: between comb
   and tacho on the degradation ladder.

7. **Microphone channel.** MAFAULDA ch7. Coherence ratio on acoustics
   would demonstrate contactless coherent order analysis. Low cost, good
   demo value.

8. **Order-domain cyclostationarity comparison.** Benchmark coherence
   ratio against CMS/cyclic spectral coherence (the established method for
   the same physics). If ratio matches CMS at a fraction of the compute,
   that is the publishable claim.

9. **Severity regression, not classification.** Predict imbalance mass /
   misalignment mm from coherent 1x/2x amplitude, per speed band. Tests
   whether coherent amps are better calibrated than power-avg amps.

10. **Block bootstrap CIs on the ratio itself.** Resample blocks to put a
    confidence interval on each coherence ratio; ratio_low > 0.5 becomes a
    defensible alarm rule for the product, not just a model feature.

11. **Transfer test to NanoPrecise field data.** Freeze the MAFAULDA-trained
    RATIO model, run on a handful of labeled MachineDoctor cases (Grasim,
    BICA pump) using the maven variant. The only test that matters
    commercially.

12. **Quantum footnote for the writeup.** Actual QPE offers no asymptotic
    advantage here (FFT is O(n log n) and n is small); the transferable
    content is the classical scaffolding. State this explicitly to
    preempt reviewer eye-rolls.

13. **Longer records via stitching.** MAFAULDA runs are 5 s (~100-300
    revs). Coherently stitch consecutive same-condition runs using tacho
    phase continuity to test N up to ~500 blocks and find where the sqrt(N)
    law saturates (speed drift decorrelation time).
