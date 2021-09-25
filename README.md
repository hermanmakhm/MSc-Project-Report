# MSc Project Report

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
tex2jax: {
inlineMath: [['$','$'], ['\\(','\\)']],
processEscapes: true},
jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
TeX: {
extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
equationNumbers: {
autoNumber: "AMS"
}
}
});
</script>
## File Structure Description

This is the LaTeX repository for my MSc project report titled _Semi-Empirical Optimisation of the Shape of a Surface Reducing Turbulent Skin Friction_. It contains a main.tex which pulls from other separate .tex files from the sub-folders as well as the bibliography Project.bib in the main folder. The output files, including the PDF, is located in the folder _build_. The Python script used for this project is located in the folder _python\_script_. This project may be further updated past the submission deadline of the report, therefore MSc\_Project\_Submitted\_for\_Grading.pdf and mean\_profile\_at\_submission.py were preserved as the report and Python script respectively at the time of submission.

## Abstract
This report aims to study the net power reduction that may occur with a fluid flowing past a wavy wall (WW) at an oblique angle $\theta$ to the flow as compared to a reference flat-plate channel flow. The analysis is done by comparing dissipation rates of WW and the spatial Stokes layer (SSL) via linearised boundary layer equations. By prescribing the spanwise and streamwise wavelengths of the WW walls, along with a corresponding SSL forcing amplitude that matches the spanwise WW phase averaged shear with that of SSL, we are able to make assumptions to utilise SSL data for the WW flow. This was done in [Chernyshenko (2013)](https://arxiv.org/abs/1304.4638). However, at the low Reynolds number regimes for the SSL flow from which data were sourced at a friction velocity based Reynolds number of 200, it is now believed that there may be significant differences in the mean streamwise velocity profile, and therefore changes the dissipation due to the mean profile, which was not considered in [Chernyshenko (2013)](https://arxiv.org/abs/1304.4638).

Therefore this report attempts to correct for that by estimating the mean velocity profile of WW flow with a linear and logarithmic portion in order to find the actual net power reduction achieved by WW. It was found that there is a system of equations that uniquely solves for the net power reduction for a given configuration by simply solving for the vertical displacement of the logarithmic portion of the mean profile compared to the reference flow. However, this is contingent on a proper estimate of the ratio of wall friction drag between WW and the reference flow, which is not easily defined due to the presence of pressure drag in WW. By assuming pressure drag is much smaller than total drag, we are able to close the system of equations to find that drag reduction with WW is not actually achievable but rather the WW requires a minimum of 17.15\% more power at corresponding SSL forcing amplitude of 6 with streamwise wavelength of approximately 1200 (in the reference flow wall units). However, there is compelling evidence for why this result may be incorrect; although which step/assumption made caused this error remains to be seen.
