import React, { useState } from "react";
import axios from "axios";
import Head from "next/head";
const textcheck = () => {
  const [result, setresult] = useState("");
  const [isSensitive, setisSensitive] = useState("");
  const [input, setinput] = useState("");
  let resultant;

  const handleTextCheckSubmission = (e) => {
    e.preventDefault();
    axios
      .get(process.env.NEXT_PUBLIC_BASE_URL + "/textclass?needy=" + input)
      .then((response) => {
        console.log(response);
        setresult(response.data[1]);
        setisSensitive(response.data[0])
      })
      .catch((err) => {
        console.log(err);
      });
    
  };
  return (
    <>
      <Head>
        <title>Text Sensitivy Check</title>
      </Head>
      <section className="text-gray-600 body-font">
        <div className="container mx-auto flex px-5 py-24 md:flex-row flex-col items-center">
          <div className="lg:flex-grow md:w-1/2 lg:pr-24 md:pr-16 flex flex-col md:items-start md:text-left mb-16 md:mb-0 items-center text-center">
            <h1 className="title-font sm:text-4xl text-3xl mb-4 font-medium text-gray-900">
              Enter the text you need to check for sensitivity
            </h1>
            <p className="mb-8 leading-relaxed">
              This will send a request to our custom made fast api and predict
              the result from our tenserflow model.
            </p>
            <form onSubmit={(e) => handleTextCheckSubmission(e)}>
              <div className="flex w-full md:justify-start justify-center items-end">
                <div className="relative mr-4 md:w-full lg:w-full xl:w-1/2 w-2/4">
                  <label className="leading-7 text-sm text-gray-600">
                    Enter the text
                  </label>
                  <input
                    value={input}
                    onChange={(e) => {
                      setinput(e.target.value);
                    }}
                    type="text"
                    id="hero-field"
                    name="hero-field"
                    className="w-full bg-gray-100 rounded border bg-opacity-50 border-gray-300 focus:ring-2 focus:ring-indigo-200 focus:bg-transparent focus:border-indigo-500 text-base outline-none text-gray-700 py-1 px-3 leading-8 transition-colors duration-200 ease-in-out"
                    required
                  />
                </div>
                <button
                  type="submit"
                  className="inline-flex text-white bg-indigo-500 border-0 py-2 px-6 focus:outline-none hover:bg-indigo-600 rounded text-lg"
                >
                  Check for Sensitivity
                </button>
              </div>
            </form>

            <p className="text-sm mt-2 text-gray-500 mb-8 w-full">
              The result would be displayed on the right
            </p>
          </div>
          {isSensitive?<>
            <div className="text-4xl text-red-600 lg:max-w-lg lg:w-full md:w-1/2 w-5/6">
            {result}
          </div>

          </>:<>
          <div className="text-4xl text-green-600 lg:max-w-lg lg:w-full md:w-1/2 w-5/6">
            {result}
          </div>
</>}
        </div>
      </section>
    </>
  );
};

export default textcheck;
