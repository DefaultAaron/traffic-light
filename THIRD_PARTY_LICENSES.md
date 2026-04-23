# Third-Party Licenses

This project vendors source code from the following third-party projects.
Each vendored file carries an upstream attribution header; this file
records the project-level license text and provenance.

---

## ByteTrack

- **Upstream**: https://github.com/ifzhang/ByteTrack
- **License**: MIT
- **Copyright**: Copyright (c) 2021 Yifu Zhang
- **Pinned commit**: `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`
- **Vendored at**: `inference/tracker/` — files `basetrack.py`,
  `byte_tracker.py`, `kalman_filter.py`, `matching.py`.
- **Local changes**: removed `torch` and YOLOX-specific imports; replaced
  `cython_bbox` with numpy IoU; replaced `lap.lapjv` with
  `scipy.optimize.linear_sum_assignment`; `BYTETracker` now takes explicit
  keyword arguments; added `reset()`. Details in each file's docstring.

### MIT License (ByteTrack)

```
MIT License

Copyright (c) 2021 Yifu Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
