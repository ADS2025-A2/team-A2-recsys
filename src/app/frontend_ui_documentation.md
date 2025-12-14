# Frontend Design & UI Documentation

This document describes the **Front-End design structure**, visual guidelines, CSS styles, and UI patterns used across the **four main Streamlit files**:

* `Home.py`
* `1_Details.py`
* `2_Watchlist.py`
* `3_Profile.py`

The goal is to allow **any developer** to easily understand, maintain, and extend the UI without breaking the visual identity.

---

## 1. Global Visual Identity

### 🎨 Color Palette

| Usage                         | Color      | Hex       |
| ----------------------------- | ---------- | --------- |
| Main background               | Dark black | `#111111` |
| Primary color | Red        | `#d72a18` |
| Hover red                     | Dark red   | `#b02114` |
| Primary text                  | White      | `#FFFFFF` |
| Secondary text                | Gray       | `#AAAAAA` |
| Alert background              | Dark gray  | `#2b2b2b` |

> **Key rule:** never use pure white as the main background.

---

## 2. Typography

### Fonts Used

* **Main titles**: `Helvetica Neue`, `Arial`, sans-serif
* **Special titles (Home)**: `Bebas Neue`
* **Body text**: `Arial`, `Montserrat`, sans-serif

### Standard Font Sizes

| Element           | Size          |
| ----------------- | ------------- |
| Main title (Home) | `70px – 80px` |
| Section title     | `48px – 60px` |
| Subtitles         | `20px – 24px` |
| Body text         | `16px – 18px` |
| Input labels      | `15px – 16px` |

---

## 3. General Layout

### 🧱 Base Structure

* Full dark background (`stAppViewContainer`)
* Sidebar visible **except on login**
* Content aligned using `st.columns`

```css
[data-testid="stAppViewContainer"] {
  background-color: #111111;
}
```

---

## 4. Sidebar (Global)

### Style

* Background: `#111111`
* Text: white
* Active item: Red

```css
[data-testid="stSidebar"] {
  background-color: #111111;
}

[data-testid="stSidebarNav"] span {
  color: #FFFFFF;
  font-size: 16px;
}

[data-testid="stSidebarNav"] .css-1fv8s86 {
  color: #d72a18;
}
```

### Rules

* Sidebar **hidden on login screen**
* Visible only when the user is authenticated

---

## 5. Buttons

### Standard Button Style

* Red background
* White or black text (depending on context)
* Rounded corners
* Hover scale animation

```css
div[data-testid="stButton"] button {
  background-color: #d72a18;
  border-radius: 8px;
  padding: 8px 20px;
  font-weight: bold;
}

div[data-testid="stButton"] button:hover {
  background-color: #b02114;
  transform: scale(1.02);
}
```

### Conventions

* Primary actions → red
* Secondary actions → same style (consistency)

---

## 6. Inputs & Multiselect

### TextInput

* White background on login
* Dark background inside the app

```css
div[data-testid="stTextInput"] input {
  background-color: #111;
  color: #fff;
  border-radius: 6px;
}
```

### Multiselect / Select

* Dark background
* White border
* Red hover on options
* Red selected chips

```css
div[data-baseweb="select"] > div {
  background-color: #111;
  border-radius: 8px;
}

ul li:hover {
  background-color: #d72a18;
}
```

---

## 7. Login Screen (Home.py)

### Login Box

* Centered layout
* Dark background
* Rounded corners
* Large red title

```css
.login-box {
  max-width: 600px;
  padding: 50px;
  border-radius: 15px;
}
```

### Rules

* Sidebar hidden
* White inputs
* Centered red button

---

## 8. Home – Recommendations

### Animated Titles

* Fade-in animation
* Red glow effect

Classes used:

* `.red-title`
* `.red-subtitle`

---

## 9. Movie Cards / Forms

### Design

* Rounded borders
* Internal padding
* Column-based layout

```html
<div style="padding:10px; border-radius:10px;">
```

---

## 10. Details Page

### Layout

* Two-column layout:

  * Left: movie information
  * Right: poster + rating

### Poster

* Centered
* Rounded corners
* Width: `250px`

---

## 11. Star Rating Component

* Dark theme enabled (`dark_theme=True`)
* Sizes:

  * Home: `16`
  * Details: `22`

---

## 12. Alerts (Success / Warning / Error)

```css
div[data-testid="stAlert"] {
  background-color: #2b2b2b;
  border-left: 5px solid #d72a18;
  border-radius: 8px;
}
```

---

## 13. Watchlist

* Vertical list layout
* Poster displayed on the right
* Clear actions:

  * More Information
  * Remove from Watchlist

---

## 14. Profile

### Layout

* Two-column structure:

  * Left: user info & preferences
  * Right: logout button

### Preferences

* Multiselect component
* Maximum of 5 selections

