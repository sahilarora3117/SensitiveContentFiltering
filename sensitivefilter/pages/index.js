import Head from "next/head";
import Link from "next/link";
export default function Home() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen py-2">
      <Head>
        <title>Sensitive Content Filter</title>
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <main className="flex flex-col items-center justify-center w-full flex-1 px-20 text-center">
        <h1 className="text-6xl font-bold">
          Welcome to{" "}
          <a className="text-blue-600" href="https://nextjs.org">
            Sensitive Content Checker
          </a>
        </h1>

        <p className="mt-3 text-2xl">Try out our sensitivity checkers... </p>

        <div className="flex flex-wrap items-center justify-around max-w-4xl mt-6 sm:w-full">
          <Link href="/textcheck">
            <h3 className="cursor-pointer text-2xl font-bold">
              Text Sensitivity Checking &rarr;
            </h3>
          </Link>

          <Link href="/imagecheck">
            <h3 className="cursor-pointer text-2xl font-bold">
              Image Sensitivity Checking &rarr;
            </h3>
          </Link>
        </div>
      </main>

      <footer className="flex items-center justify-center w-full h-24 border-t">
          Made Using Tensorflow, FastAPI, NextJS
      </footer>
    </div>
  );
}
