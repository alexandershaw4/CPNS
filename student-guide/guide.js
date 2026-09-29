(() => {
  "use strict";

  const byId = id => document.getElementById(id);
  const prior = byId("prior");
  const hit = byId("hit");
  const falseAlarm = byId("false-alarm");
  const posterior = byId("posterior-value");
  const posteriorBar = byId("posterior-bar");
  const arithmetic = byId("bayes-arithmetic");

  function formatCount(value) {
    return Number.isInteger(value) ? String(value) : value.toFixed(1);
  }

  function updateBayes() {
    const baseRate = Number(prior.value) / 100;
    const sensitivity = Number(hit.value) / 100;
    const falsePositiveRate = Number(falseAlarm.value) / 100;
    const genuine = 100 * baseRate * sensitivity;
    const falsePositives = 100 * (1 - baseRate) * falsePositiveRate;
    const probability = 100 * genuine / (genuine + falsePositives);
    byId("prior-value").textContent = prior.value + "%";
    byId("hit-value").textContent = hit.value + "%";
    byId("false-alarm-value").textContent = falseAlarm.value + "%";
    posterior.textContent = Math.round(probability) + "%";
    posteriorBar.style.width = probability + "%";
    posteriorBar.parentElement.setAttribute("aria-label", "Probability that a tone was present: " + Math.round(probability) + "%");
    arithmetic.textContent = formatCount(genuine) + " genuine detections ÷ (" + formatCount(genuine) + " genuine + " + formatCount(falsePositives) + " false alarms) ≈ " + Math.round(probability) + " in 100";
  }

  [prior, hit, falseAlarm].forEach(input => input.addEventListener("input", updateBayes));
  updateBayes();

  const frequency = byId("frequency");
  const width = byId("width");
  const amplitude = byId("amplitude");
  const target = { frequency: 12, width: 2.4, amplitude: 0.9 };

  function spectrum(f, settings, observed) {
    const baseline = 0.09 + 0.04 * Math.exp(-f / 10);
    const peak = settings.amplitude / (1 + Math.pow((f - settings.frequency) / settings.width, 2));
    const smallIrregularity = observed ? 0.009 * Math.sin(f * 3.7) + 0.005 * Math.cos(f * 7.1) : 0;
    return baseline + peak + smallIrregularity;
  }

  function toPath(settings, observed) {
    let path = "";
    for (let index = 0; index <= 140; index++) {
      const f = 2 + 28 * index / 140;
      const value = spectrum(f, settings, observed);
      const x = 52 + (f - 2) / 28 * 564;
      const y = 270 - value / 1.4 * 220;
      path += (index ? " L" : "M") + x.toFixed(1) + " " + y.toFixed(1);
    }
    return path;
  }

  byId("observed-curve").setAttribute("d", toPath(target, true));

  function updateSpectrum() {
    const model = {
      frequency: Number(frequency.value),
      width: Number(width.value),
      amplitude: Number(amplitude.value)
    };
    byId("frequency-value").textContent = model.frequency.toFixed(1) + " Hz";
    byId("width-value").textContent = model.width.toFixed(1);
    byId("amplitude-value").textContent = model.amplitude.toFixed(2);
    byId("model-curve").setAttribute("d", toPath(model, false));

    let sumSquaredError = 0;
    for (let i = 0; i <= 140; i++) {
      const f = 2 + 28 * i / 140;
      const difference = spectrum(f, target, true) - spectrum(f, model, false);
      sumSquaredError += difference * difference;
    }
    const mismatch = Math.sqrt(sumSquaredError / 141);
    byId("fit-error").textContent = (mismatch * 100).toFixed(1);
    byId("fit-feedback").textContent = mismatch < 0.025
      ? "Very close fit. You recovered the shape, but this alone cannot establish a brain mechanism."
      : mismatch < 0.085
        ? "Getting close. Check the position and width of the peak."
        : "The curves still disagree. Try the peak position first, then its width and height.";
    byId("spectrum-desc").textContent = "The observed spectrum has a peak near 12 hertz. The current model peaks at " +
      model.frequency.toFixed(1) + " hertz, with a curve mismatch of " + (mismatch * 100).toFixed(1) + ".";
  }

  [frequency, width, amplitude].forEach(input => input.addEventListener("input", updateSpectrum));
  byId("reveal-fit").addEventListener("click", () => {
    const answer = byId("fit-answer");
    answer.hidden = !answer.hidden;
    byId("reveal-fit").textContent = answer.hidden ? "Reveal the generating settings" : "Hide the generating settings";
  });
  updateSpectrum();

  if ("IntersectionObserver" in window) {
    const links = [...document.querySelectorAll(".chapter-rail a")];
    const observer = new IntersectionObserver(entries => {
      const visible = entries.filter(entry => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!visible) return;
      links.forEach(link => {
        const active = link.getAttribute("href") === "#" + visible.target.id;
        link.classList.toggle("active", active);
        if (active) link.setAttribute("aria-current", "location");
        else link.removeAttribute("aria-current");
      });
    }, { rootMargin: "-15% 0px -65% 0px", threshold: [0, 0.1, 0.2] });
    document.querySelectorAll(".chapter").forEach(chapter => observer.observe(chapter));
  }
})();
