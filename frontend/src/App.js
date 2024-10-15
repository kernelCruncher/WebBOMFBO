// Importing modules
import React, { useState, useEffect } from "react";
import "./App.css";

function App() {
	const [data, setData] = useState([]);
	// Using useEffect for single rendering
	useEffect(() => {
		fetch('/uploads').then(res => res.json()).then(data => {
			console.log(data)
			setData(data);
		});
	  }, []);

	return (
		<div className="App">
			<header className="App-header">
				<h3>Multi-fidelity Bayesian Optimization</h3>
			</header>
			<div>I am hungry. Give me some files!</div>	
			<form method="POST" action="/upload" encType="multipart/form-data">
      			<p><input type="file" name="file"></input></p>
      			<p><input type="submit" value="Submit"></input></p>
    		</form>
			{
						data.map(function(item, i) {
						  return(
							<div key={i}>
							  <span>{item}</span>
							</div>
						 )
})
			}
		</div>
	);
}

export default App;
