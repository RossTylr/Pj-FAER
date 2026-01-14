/**
 * Streamlit Component Entry Point
 *
 * This file bootstraps the React schematic component for use with Streamlit.
 */

import React from "react";
import ReactDOM from "react-dom/client";
import { withStreamlitConnection } from "streamlit-component-lib";
import Schematic from "./Schematic";

const WrappedSchematic = withStreamlitConnection(Schematic);

const root = ReactDOM.createRoot(document.getElementById("root") as HTMLElement);
root.render(
  <React.StrictMode>
    <WrappedSchematic />
  </React.StrictMode>
);
