with import <nixpkgs> {};

stdenv.mkDerivation{
  name = "pcnn";

  src = "./"; 

  propagatedBuildInputs = [
    cudatoolkit
    (pkgs.opencv3.override {
      enableGtk2 = true;
      enableGtk3 = true;
      enableCuda = true;
    })
    gcc5
    gtk2
    gtk3
    pkgconfig
  ];
  nativeBuildInputs = [ cmake ];
}
