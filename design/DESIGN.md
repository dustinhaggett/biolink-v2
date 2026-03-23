# Design System Strategy: Clinical Precision & Tonal Depth

## 1. Overview & Creative North Star
**The Creative North Star: "The Digital Laboratory"**

This design system moves away from the "SaaS template" aesthetic to create an environment that feels like a high-end medical research facility: sterile but not cold, authoritative but human-centric. We achieve this by rejecting the traditional "box-and-border" layout in favor of **Tonal Layering** and **Editorial Asymmetry**. 

The system utilizes a sophisticated interplay between a "Manrope" display scale (industrial yet modern) and an "Inter" functional scale (highly legible and clinical). By leaning into whitespace and surface-on-surface depth, we communicate the precision required for drug discovery without the visual clutter of legacy medical software.

---

## 2. Colors & Surface Architecture

### The "No-Line" Rule
Standard 1px borders are strictly prohibited for sectioning. They create visual noise that distracts from complex data. Instead, define boundaries through:
- **Background Shifts:** Use `surface_container_low` against a `background` base to define regions.
- **Tonal Transitions:** Use a 2px vertical "accent bar" of `primary` to denote active sections rather than boxing them in.

### Surface Hierarchy & Nesting
Think of the UI as physical layers of frosted glass and premium cardstock.
*   **Base Layer:** `surface` (#f8fafb) for the global canvas.
*   **Mid-Tier:** `surface_container` (#eceeef) for sidebars or secondary navigation.
*   **High-Tier (Content):** `surface_container_lowest` (#ffffff) for the primary data cards. 
*   **Nesting Strategy:** An inner data table should sit on `surface_container_low`, nested within a `surface_container_highest` module to create natural, "soft-focus" depth.

### The "Glass & Gradient" Rule
To elevate "Clinical" to "Premium," use glassmorphism for floating overlays (e.g., Command Palettes or Tooltips).
- **Glass Token:** Use `surface_container_lowest` at 85% opacity with a `24px` backdrop blur.
- **Signature Gradient:** For primary CTAs and Hero headers, use a subtle linear gradient: `primary` (#00606d) to `primary_container` (#267987) at a 135° angle. This adds "visual soul" and depth.

---

## 3. Typography: The Editorial Scale

We use a dual-font system to balance authority with utility.

| Category | Typeface | Token | Use Case |
| :--- | :--- | :--- | :--- |
| **Display** | Manrope | `display-lg` (3.5rem) | High-level data synthesis / Dashboards |
| **Headline** | Manrope | `headline-sm` (1.5rem) | Research module titles |
| **Title** | Inter | `title-md` (1.125rem) | Card headers / Section titles |
| **Body** | Inter | `body-md` (0.875rem) | Scientific descriptions / Metadata |
| **Label** | Inter | `label-sm` (0.6875rem) | Confidence scores / Micro-copy |

**Editorial Intent:** Use `display-md` with `on_surface_variant` for background titles to create a "watermark" effect, grounding the page in a sophisticated, editorial layout.

---

## 4. Elevation & Depth

### The Layering Principle
Avoid "Drop Shadows" whenever possible. Achieve lift by placing a `surface_container_lowest` element on a `surface_dim` background. This creates a "Paper-on-Stone" effect that feels tactile and high-end.

### Ambient Shadows
Where floating depth is required (e.g., Modals), use an **Ambient Tinted Shadow**:
- **Values:** `0px 20px 40px`
- **Color:** `on_surface` (#191c1d) at 6% opacity.
- **Result:** A soft, natural lift that mimics diffused laboratory lighting.

### The "Ghost Border" Fallback
If a container requires a boundary for accessibility (e.g., Input Fields), use a **Ghost Border**:
- **Token:** `outline_variant` (#bdc9ca) at 30% opacity. Never use 100% opacity.

---

## 5. Components

### Buttons
- **Primary:** Gradient (`primary` to `primary_container`), white text, `round_md` (0.375rem).
- **Secondary:** `surface_container_high` with `on_surface`. No border.
- **Tertiary:** Text-only using `primary` with a `2.5` spacing (0.85rem) horizontal padding for a wide, premium hit-area.

### Cards & Data Modules
- **Rule:** Absolute prohibition of divider lines.
- **Spacing:** Use `spacing.6` (2rem) between internal card sections.
- **Separation:** Use a 4px left-side border-radius accent in `tertiary` to categorize research types.

### Research Confidence Chips
- **High (Green):** `on_tertiary_container` on `tertiary_fixed`.
- **Medium (Yellow):** Custom amber-tinted surface (use `secondary_container`).
- **Low (Red):** `on_error_container` on `error_container`.
- **Style:** Semi-pill shape (`round_xl`).

### Medical Data Inputs
- **State:** Active inputs use a `primary` ghost-border (20% opacity) and a `primary_fixed` subtle glow.
- **Feedback:** Error states use `error` text with a `surface_container_lowest` background to ensure high-contrast readability against the global `surface`.

### Specialized Component: The "Molecule Inspector"
A floating glass container (`surface_container_lowest` @ 90% opacity) used for inspecting chemical structures. It uses `shadow_xl` and no borders, appearing to float above the research grid.

---

## 6. Do’s and Don’ts

### Do
*   **Do** use `spacing.12` (4rem) for hero-to-content transitions to let the "clinical" whitespace breathe.
*   **Do** use `manrope` for any number-heavy displays (Confidence scores, molecular weights) to leverage its modern, tabular feel.
*   **Do** nest `surface_container_lowest` cards inside `surface_container_low` sections to create soft hierarchy.

### Don’t
*   **Don't** use 1px solid borders for layout separation; it looks like "out-of-the-box" Bootstrap.
*   **Don't** use pure black (#000) for text. Always use `on_surface` (#191c1d) to maintain a premium, soft-contrast feel.
*   **Don't** use standard "drop shadows" with 20%+ opacity. They feel heavy and "dirty" in a sterile research environment.