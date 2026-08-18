# Technical Specification: Smart AI Window/Fence Bird Feeder (v2)

**Revision notes:** This supersedes the original blueprint. Key changes: local species classification replaces the cloud vision API, a two-stage bird-confirmation pipeline is retained (not simplified) due to high-traffic false-positive risk, camera/lens choice is updated, mounting location is now open between window and fence, and the "dual-purpose security camera" concept has been dropped.

---

## 1. Project Overview

An AI-powered smart bird feeder with an embedded or window-adjacent camera. The system uses a two-stage local vision pipeline (motion gate + object detection) to confirm a bird is present, selects the best captured frame during that visit, runs local on-device species classification, logs stats to a database, and serves a local web dashboard over Wi-Fi.

**Explicitly out of scope (descoped from earlier discussion):** dual-purpose security/surveillance functionality (person/vehicle detection, night-time coverage, multi-camera security alerting). This build is bird-focused only.

---

## 2. Hardware Architecture & Bill of Materials

- **Compute Unit:** Raspberry Pi 3 Model B (v1.2) — 1.2 GHz Quad-Core ARM Cortex-A53, 1 GB LPDDR2 RAM, onboard Wi-Fi + Ethernet.
- **Camera Module:** Raspberry Pi Camera Module 3 (standard field of view, *not* the wide variant), 12MP Sony IMX708 sensor, phase-detect autofocus.
  - **Why not wide-angle:** the camera position is fixed and known relative to the feeding spot (embedded in the feeder or window-mounted facing it), so a standard FOV keeps more native sensor resolution on the bird itself rather than surrounding scenery — important since the species classifier performs better on a well-filled, cropped subject.
  - **Why autofocus matters here:** older fixed-focus modules (V1/V2) required subjects at a fairly specific distance to stay sharp. Autofocus removes this failure mode for a feeder where birds move around at variable close range.
  - **Connector check:** confirm the cable ships with (or includes an adapter for) the 15-pin CSI connector used by Pi 3B — Module 3 often ships configured for the newer 22-pin connector used on Pi 5/Zero.
  - **Open item:** verify the module's minimum focus distance against the actual camera-to-feeding-perch distance if the camera ends up embedded very close to the perch.
- **Power Supply:** 5V / 2.5A+ Micro-USB wall adapter — **only valid if the Pi/camera end up window-side with easy indoor power access.** If mounted on the fence, see Section 6 (Mounting Decision) for the power implications.
- **Physical Housing:** Custom 3D-printed enclosure (PETG or ASA), weatherproof seed tray with drainage, dry electronics chamber sealed from the feeding chamber, protective optical dome over the camera aperture.

---

## 3. System Architecture & Vision Pipeline

```
[ Camera Stream (CSI / OpenCV) ]
              │
              ▼
    [ Stage 1: Local Motion Filter ]  ──(No Motion)──► Drop Frame / Idle
              │ (Motion Exceeds Threshold)
              ▼
    [ Stage 2: Local Object Detector ] ──("Not a bird")─► Drop Frame
              │ (TFLite MobileNetV2-SSD or YOLOv8 Nano detects "bird" bounding box)
              ▼
   [ Stage 3: Visit Session Tracking ]
              │ (IoU-based bounding box tracking — see Section 4)
              ▼
   [ Stage 4: Best-Frame Selection within session ]
              │ (sharpness + bbox size/position scoring — see Section 4)
              ▼
   [ Stage 5: Local Species Classifier ] ──► TFLite bird species model (964 classes)
              │ (returns species label + confidence)
              ▼
   [ Confidence Threshold Check ] ──(low confidence)──► Log as "unknown", skip/flag
              │ (high confidence)
              ▼
   [ SQLite Database Write & Asset Save ]
              │
              ▼
 [ Local Web Dashboard (Flask/FastAPI) ]
```

### Why the two-stage approach was kept (not simplified to motion → classifier)
The species classifier has no "not a bird" output — it will confidently assign one of its 964 species labels to *any* input, including squirrels, wind-blown branches, or shadows. In a high-traffic feeder location, this would flood the dashboard/stats with false "sightings." The object-detection gate (Stage 2) is retained specifically to filter non-bird motion before it reaches the classifier. A confidence threshold on the classifier's output is an additional safety net, not a replacement for this gate.

### Stage breakdown
- **Stage 1 — Motion Gate:** Low-res (320×240) frame differencing, keeps baseline CPU usage low.
- **Stage 2 — Object Detection Gate:** Quantized TFLite model (MobileNetV2-SSD or YOLOv8 Nano) confirms a "bird" class bounding box at >70% confidence before proceeding. Expect ~600ms–1.2s inference on Pi 3B CPU (revised up from original 200–400ms estimate, which assumed hardware acceleration not present on a bare 3B).
- **Stage 3 — Visit Tracking:** See Section 4 — determines whether consecutive detections represent one continuous visit or a new arrival.
- **Stage 4 — Best-Frame Selection:** See Section 4 — captures multiple candidate frames per visit and scores them rather than relying on a single snapshot.
- **Stage 5 — Species Classification (now local, not cloud):** TFLite model based on Google's AIY `birds_V1/3` (MobileNet architecture, trained on iNaturalist data, 964 species classes). Runs once per visit, on the selected best frame.
  - **Why local instead of the original cloud LLM call (Gemini/GPT-4o mini):** removes per-call API cost and network dependency; a dedicated fine-tuned bird classifier is also generally more accurate on fine-grained species distinctions than a general-purpose vision LLM.
  - **Regional coverage:** common Atlanta-area backyard species (Northern Cardinal, Carolina Chickadee, Carolina Wren, Tufted Titmouse, Mourning Dove, Blue Jay, Downy/Red-bellied Woodpecker, American Goldfinch, White-breasted Nuthatch, European Starling, American Crow, etc.) are mainstream North American species well-represented in iNaturalist data and should be covered by the 964-class label set. **Action item before finalizing:** pull the model's label map and cross-check against the expected local species list, and spot-test with reference photos of the top 15–20 expected visitors to confirm accuracy, particularly on look-alike pairs (e.g., House Finch vs. Purple Finch).
  - **Optional fallback:** cloud vision API call retained as an optional secondary path only for low-confidence local classifications, not as the primary identification method.
- **Confidence Threshold:** Only log as a confirmed species sighting above a chosen threshold (e.g., 85–90%); below that, log as "unknown/unclassified" rather than a likely-wrong species label.

---

## 4. Visit Detection: Best-Frame Capture & Same-Bird Tracking

### Capturing the best frame per visit
Rather than a single snapshot at first detection (often blurred or mid-motion):
- Once Stage 2 confirms a bird bounding box, continue capturing candidate frames every ~200–300ms for a short window.
- Score each candidate on:
  - **Sharpness** — Laplacian variance (`cv2.Laplacian(img, cv2.CV_64F).var()`); higher = sharper.
  - **Bounding box size** — prefer frames where the bird fills more of the frame.
  - **Bounding box position** — prefer frames where the bird isn't clipped at frame edges.
- Run the species classifier once, on the single winning frame, to keep inference cost equivalent to a single-shot design.

### Determining "same bird still present" vs. "new visit"
Full individual bird re-identification is unnecessary and impractical; a lightweight tracking approach is sufficient:
- **Frame-to-frame IoU tracking:** compare each new detection's bounding box to the prior frame's. High overlap + similar size ⇒ treat as continuous presence, not a new visit.
- **Session concept:** a "visit" is a continuous tracked presence. The session ends after a grace period (e.g., 3–5 seconds) of no detection — this doubles as the cooldown/debounce mechanism from the original design, now driven by tracking loss rather than a flat timer alone.
- **High-traffic caveat:** with frequent bird turnover, a departing bird and an arriving bird within the grace window can be merged into one session by a simple tracker. Mitigations:
  - Shorten the grace period to reduce merge risk.
  - Consider basic multi-object tracking (e.g., simple centroid tracking) if multiple birds are frequently present simultaneously, so concurrent individuals aren't conflated into a single session.

---

## 5. Software Stack & Local Web Application

*(Unchanged from original, included for completeness)*
- **OS:** Raspberry Pi OS 64-bit (Lite or Desktop).
- **Language/Runtime:** Python 3.10+ in a venv.
- **Key Libraries:** opencv-python, tflite-runtime, sqlite3, Flask (or FastAPI). (`httpx`/`requests` now optional, only needed if the cloud fallback path is implemented.)
- **Dashboard:** Tailwind CSS + Chart.js. Live feed, gallery view (species/timestamp/confidence), analytics (frequency, peak hours), settings panel (thresholds, cooldown, model management).

---

## 6. Mounting Decision — Open Item

The original blueprint assumed suction-cup window mounting. This is now **undecided between two options**, pending two checks:

**Option A — Window-mounted (original approach), feeder facing the window:**
- Preserves direct human sightline (you see birds with your own eyes, not just via dashboard).
- Simple power (indoor Micro-USB adapter, no weatherproof cable run).
- Retains original one-way mirror film consideration (hides indoor movement from birds through the glass).

**Option B — Fence-mounted, camera/Pi embedded in the feeder:**
- Camera-to-bird distance likely very short and fixed — reinforces standard FOV lens choice; verify autofocus close-focus range.
- **Power becomes a significant added design problem:** requires either a weatherproof cable run to the house or battery + solar (capacity/charge-controller sizing not yet scoped).
- **Weatherproofing requirements increase substantially** — full outdoor exposure vs. window-adjacent placement.
- **If the feeder faces away from the house/window, direct human viewing is lost** — bird viewing becomes camera/dashboard-mediated only. If seeing birds directly through the window remains a goal, feeder orientation (feeding side facing the window) matters independent of where the camera itself sits.
- **Wi-Fi range is unverified** — typical outdoor router range is roughly 100–150 ft line-of-sight, reduced by walls/fencing/foliage. **Action item: physically test Wi-Fi signal strength at the fence location before committing to this option.** If signal is marginal, options include a mesh/extender near the window, a directional antenna on the Pi, or running Ethernet (Pi 3B has onboard Ethernet) if a cable run is feasible.

**Recommendation:** resolve the Wi-Fi test and the "do you want direct sightline or camera-only viewing" question before finalizing mechanical/CAD design, since they materially change the power and housing requirements.

---

## 7. Considered and Rejected: Frigate-Based Architecture

Frigate + a species-classifier sidecar (e.g., the WhosAtMyFeeder project pattern) was evaluated as an alternative pipeline. **Decision: not adopted for this build.**
- Frigate requires a full NVR-style deployment (MQTT broker, continuous decode) and, per current guidance, is best run on an N100 mini PC or Pi 5 + Hailo accelerator rather than a bare Pi 3B — the Coral USB TPU previously the standard recommendation is now deprecated (driver archived, unsupported on modern kernels).
- Since the security/dual-purpose use case was dropped, the main benefit Frigate would have added (continuous recording, general object/person detection, clip retention) is no longer needed.
- The useful piece of that research — the AIY `birds_V1/3` TFLite species classifier — has been pulled directly into this pipeline's Stage 5 without adopting Frigate itself, keeping the Pi 3B viable.

---

## 8. Mechanical & Thermal Design Requirements

*(Largely unchanged from original; weatherproofing emphasis increases if Option B/fence-mounting is selected — see Section 6.)*
- Enclosure sized for Pi 3B footprint (85×56mm) plus port clearances.
- Passive ventilation near CPU heat zones.
- Sealed separation between electronics chamber and seed/feeding chamber.
- Protective optical dome over camera aperture, sealed against humidity/dust.

---

## 9. Summary of Open Action Items
1. Confirm Module 3 camera cable/connector matches Pi 3B (15-pin) or obtain correct adapter.
2. Verify camera close-focus distance against actual embedded mounting distance (if Option B).
3. Pull the species classifier's label map and confirm coverage of expected Atlanta-area species; spot-test accuracy with reference photos.
4. Decide window vs. fence mounting (Section 6) — dependent on Wi-Fi signal test and viewing-preference decision.
5. If fence-mounted, scope power solution (weatherproof cable run vs. battery/solar) and finalize weatherproofing design.
6. Tune visit-session grace period and confidence thresholds empirically once hardware is running, to balance false positives (junk sightings) against missed/split visits.